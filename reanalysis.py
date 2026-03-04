#!/usr/bin/env python3
"""
Reanalysis of PICFLU Cohort BAL Data (Boyd et al. 2020, Nature)

Applies multiple normalization methods to real BAL biomarker data from
critically ill children with influenza, demonstrating that normalization
choice materially alters which biomarkers reach statistical significance.

Data source: Boyd et al., Nature 587:466-471, 2020 (PMID 33116313)
Supplementary Table 3: Cytokine/protein measurements in lower respiratory
tract fluid from the PICFLU cohort.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from biomarker_dilution_sim import (
    normalize_data, centered_log_ratio,
    multiple_testing_correction,
    compute_distance_matrix, permanova_r2,
    pca_centroid_analysis,
)

# Output directories
FIGURES_DIR = Path('figures')
RESULTS_DIR = Path('results')
DATA_DIR = Path('manuscript_latex/data')
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Publication-quality plotting
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'savefig.bbox': 'tight',
})


def load_picflu_data():
    """Load and clean PICFLU BAL data from Boyd et al. 2020."""
    df = pd.read_excel(DATA_DIR / 'Boyd2020_MOESM4.xlsx',
                       sheet_name='Supplemental Table 3')

    # Define analyte columns (starting from Total Protein)
    analyte_start = df.columns.get_loc('Total Protein')
    all_analyte_cols = list(df.columns[analyte_start:])

    # Handle censored values (> and < prefixed strings)
    for col in all_analyte_cols:
        df[col] = df[col].apply(_parse_censored)

    # Convert to numeric
    for col in all_analyte_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select analytes with sufficient data (>= 40 non-missing values)
    good_analytes = [c for c in all_analyte_cols
                     if df[c].notna().sum() >= 40 and c != 'Total Protein']

    print(f"Loaded PICFLU data: {len(df)} subjects, {len(good_analytes)} analytes with sufficient data")
    print(f"  Total Protein available: {df['Total Protein'].notna().sum()} subjects")

    # Create clinical group variable: PrAHRF or Death (most balanced split)
    df['outcome'] = (df['PrAHRF or Death'] == 'Yes').astype(int)
    print(f"  Clinical groups (PrAHRF/Death): {df['outcome'].value_counts().to_dict()}")

    return df, good_analytes


def _parse_censored(val):
    """Parse censored values like '>10000' or '<84.57'."""
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('>'):
            try:
                return float(val[1:])
            except ValueError:
                return np.nan
        elif val.startswith('<'):
            try:
                # Use LOD/sqrt(2) substitution
                return float(val[1:]) / np.sqrt(2)
            except ValueError:
                return np.nan
        else:
            try:
                return float(val)
            except ValueError:
                return np.nan
    return val


def prepare_pca_data(X, method_name, log_transform=True, scale=True):
    """Prepare data for PCA with optional log-transform and scaling.

    Parameters
    ----------
    X : np.ndarray
        Normalized data matrix (n_samples × n_features).
    method_name : str
        Normalization method name. CLR is already log-transformed, so
        log_transform is skipped for it.
    log_transform : bool
        If True, apply log(x) to non-CLR data (with pseudocount for zeros).
    scale : bool
        If True, apply StandardScaler (center to mean 0, scale to unit variance).

    Returns
    -------
    np.ndarray
        Preprocessed data ready for PCA.
    """
    X_prep = X.copy()
    if log_transform and method_name != 'clr':
        X_prep = np.maximum(X_prep, 0) + 1e-5
        X_prep = np.log(X_prep)
    if scale:
        X_prep = StandardScaler().fit_transform(X_prep)
    return X_prep


def apply_normalizations(X, analyte_names):
    """Apply all normalization methods to the data matrix.

    Returns dict of {method_name: normalized_matrix}.
    """
    methods = {}

    # 1. None (raw)
    methods['none'] = X.copy()

    # 2. Total sum
    methods['total_sum'] = normalize_data(X, method='total_sum')

    # 3. PQN
    methods['pqn'] = normalize_data(X, method='pqn')

    # 4. CLR
    methods['clr'] = normalize_data(X, method='clr')

    # 5. Median
    methods['median'] = normalize_data(X, method='median')

    # 6. Quantile
    methods['quantile'] = normalize_data(X, method='quantile')

    return methods


def differential_analysis(X, y, method_name='t_test'):
    """Perform differential analysis between two groups.

    Returns DataFrame with p-values, fold changes, and significance.
    """
    n_features = X.shape[1]
    results = []

    for j in range(n_features):
        g0 = X[y == 0, j]
        g1 = X[y == 1, j]

        # Remove NaNs
        g0 = g0[~np.isnan(g0)]
        g1 = g1[~np.isnan(g1)]

        if len(g0) < 3 or len(g1) < 3:
            results.append({
                'feature': j,
                'p_value': np.nan,
                'test_stat': np.nan,
                'mean_g0': np.nan,
                'mean_g1': np.nan,
                'fold_change': np.nan,
            })
            continue

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)

        # Fold change (use means, handle zeros)
        m0 = np.mean(g0)
        m1 = np.mean(g1)
        if m0 != 0:
            fc = m1 / m0
        else:
            fc = np.nan

        results.append({
            'feature': j,
            'p_value': p_val,
            'test_stat': t_stat,
            'mean_g0': m0,
            'mean_g1': m1,
            'fold_change': fc,
        })

    return pd.DataFrame(results)


def distance_metric_analysis(normalized, y, methods_ordered, n_permutations=999):
    """Compute PERMANOVA R² for each normalization × distance metric combination.

    Parameters
    ----------
    normalized : dict
        {method_name: normalized_matrix} from apply_normalizations().
    y : np.ndarray
        Group labels.
    methods_ordered : list
        Normalization method names in display order.
    n_permutations : int
        Number of permutations for PERMANOVA.

    Returns
    -------
    pd.DataFrame with columns: norm_method, distance_metric, permanova_r2,
        permanova_f, permanova_p
    """
    distance_metrics = ['euclidean', 'bray_curtis', 'aitchison', 'cosine',
                        'manhattan', 'canberra', 'mahalanobis']
    rows = []

    for method_name in methods_ordered:
        X_norm = normalized[method_name]

        # Drop rows with any NaN for distance computation
        valid = ~np.isnan(X_norm).any(axis=1)
        X_valid = X_norm[valid]
        y_valid = y[valid]

        if len(np.unique(y_valid)) < 2 or X_valid.shape[0] < 4:
            continue

        for metric in distance_metrics:
            # Skip invalid combinations
            if method_name == 'clr' and metric == 'bray_curtis':
                continue  # CLR produces negatives; Bray-Curtis requires non-negative
            if method_name == 'clr' and metric == 'aitchison':
                continue  # Aitchison applies CLR internally; double-CLR is meaningless
            if method_name == 'clr' and metric == 'canberra':
                continue  # Canberra requires non-negative data

            try:
                D = compute_distance_matrix(X_valid, metric)
                result = permanova_r2(D, y_valid, n_permutations=n_permutations)
                rows.append({
                    'norm_method': method_name,
                    'distance_metric': metric,
                    'permanova_r2': result['r2'],
                    'permanova_f': result['pseudo_f'],
                    'permanova_p': result['p_value'],
                })
            except Exception as e:
                print(f"  Warning: {method_name}/{metric} failed: {e}")

    return pd.DataFrame(rows)


def plot_permanova_heatmap(balf_results, fig_path, title_label,
                           blood_results=None):
    """Plot PERMANOVA R² heatmap(s) and grouped bar chart.

    Parameters
    ----------
    balf_results : pd.DataFrame
        Output of distance_metric_analysis() for BALF data.
    fig_path : str or Path
        Output file path (without extension).
    title_label : str
        Figure label, e.g. 'E18' or 'E19'.
    blood_results : pd.DataFrame or None
        If provided, adds a blood heatmap panel and comparison bar chart.
    """
    distance_metrics = ['euclidean', 'bray_curtis', 'aitchison', 'cosine',
                        'manhattan', 'canberra', 'mahalanobis']
    metric_labels = {'euclidean': 'Euclidean', 'bray_curtis': 'Bray-Curtis',
                     'aitchison': 'Aitchison', 'cosine': 'Cosine',
                     'manhattan': 'Manhattan', 'canberra': 'Canberra',
                     'mahalanobis': 'Mahalanobis'}

    def _build_pivot(df):
        pivot = df.pivot(index='norm_method', columns='distance_metric',
                         values='permanova_r2')
        # Reorder columns
        cols = [m for m in distance_metrics if m in pivot.columns]
        pivot = pivot[cols]
        pivot.columns = [metric_labels[c] for c in cols]
        return pivot

    def _build_annot(df, pivot):
        """Build annotation strings with significance stars."""
        annot = pivot.copy().astype(str)
        for _, row in df.iterrows():
            m = row['norm_method']
            metric_col = metric_labels[row['distance_metric']]
            if m in pivot.index and metric_col in pivot.columns:
                r2_val = row['permanova_r2']
                p_val = row['permanova_p']
                stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                annot.loc[m, metric_col] = f'{r2_val:.3f}{stars}'
        return annot

    has_blood = blood_results is not None and len(blood_results) > 0
    n_panels = 3 if has_blood else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))

    # Panel A: BALF PERMANOVA R² heatmap
    ax = axes[0]
    balf_pivot = _build_pivot(balf_results)
    balf_annot = _build_annot(balf_results, balf_pivot)
    sns.heatmap(balf_pivot, ax=ax, annot=balf_annot, fmt='',
                cmap='YlOrRd', vmin=0, vmax=0.5, linewidths=0.5,
                cbar_kws={'label': 'PERMANOVA R²'})
    panel_title = 'A) BALF PERMANOVA R²' if has_blood else 'A) PERMANOVA R²'
    ax.set_title(panel_title, fontweight='bold')
    ax.set_ylabel('Normalization Method')
    ax.set_xlabel('Distance Metric')

    if has_blood:
        # Panel B: Blood PERMANOVA R² heatmap
        ax = axes[1]
        blood_pivot = _build_pivot(blood_results)
        blood_annot = _build_annot(blood_results, blood_pivot)
        sns.heatmap(blood_pivot, ax=ax, annot=blood_annot, fmt='',
                    cmap='YlOrRd', vmin=0, vmax=0.5, linewidths=0.5,
                    cbar_kws={'label': 'PERMANOVA R²'})
        ax.set_title('B) Blood PERMANOVA R²', fontweight='bold')
        ax.set_ylabel('Normalization Method')
        ax.set_xlabel('Distance Metric')

        # Panel C: Grouped bar chart comparing BALF vs Blood
        ax = axes[2]
        key_methods = ['none', 'clr', 'pqn']
        x_labels = []
        balf_vals = []
        blood_vals = []
        for method in key_methods:
            for metric in distance_metrics:
                if method == 'clr' and metric in ('bray_curtis', 'aitchison', 'canberra'):
                    continue
                label = f'{method}\n{metric_labels[metric]}'
                x_labels.append(label)
                balf_row = balf_results[
                    (balf_results['norm_method'] == method) &
                    (balf_results['distance_metric'] == metric)]
                blood_row = blood_results[
                    (blood_results['norm_method'] == method) &
                    (blood_results['distance_metric'] == metric)]
                balf_vals.append(balf_row['permanova_r2'].values[0]
                                 if len(balf_row) else 0)
                blood_vals.append(blood_row['permanova_r2'].values[0]
                                  if len(blood_row) else 0)

        x = np.arange(len(x_labels))
        width = 0.35
        ax.bar(x - width/2, balf_vals, width, label='BALF', color='#D55E00',
               alpha=0.8)
        ax.bar(x + width/2, blood_vals, width, label='Blood', color='#0072B2',
               alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('PERMANOVA R²')
        ax.set_title('C) BALF vs Blood R²', fontweight='bold')
        ax.legend(fontsize=9)
    else:
        # Panel B: Grouped bar chart for key normalizations
        ax = axes[1]
        key_methods = ['none', 'clr', 'pqn']
        bar_data = {}
        for metric in distance_metrics:
            bar_data[metric_labels[metric]] = []
            for method in key_methods:
                if method == 'clr' and metric in ('bray_curtis', 'aitchison', 'canberra'):
                    bar_data[metric_labels[metric]].append(0)
                    continue
                row = balf_results[
                    (balf_results['norm_method'] == method) &
                    (balf_results['distance_metric'] == metric)]
                bar_data[metric_labels[metric]].append(
                    row['permanova_r2'].values[0] if len(row) else 0)

        x = np.arange(len(key_methods))
        n_metrics_bar = len(bar_data)
        width = 0.8 / max(n_metrics_bar, 1)
        colors = ['#E69F00', '#56B4E9', '#009E73', '#CC79A7',
                  '#D55E00', '#0072B2', '#F0E442']
        for i, (metric_name, vals) in enumerate(bar_data.items()):
            ax.bar(x + i * width, vals, width, label=metric_name,
                   color=colors[i % len(colors)], alpha=0.8)
        ax.set_xticks(x + width * n_metrics_bar / 2)
        ax.set_xticklabels(key_methods)
        ax.set_ylabel('PERMANOVA R²')
        ax.set_xlabel('Normalization Method')
        ax.set_title('B) R² by Distance Metric', fontweight='bold')
        ax.legend(fontsize=9)

    fig.suptitle(f'Figure {title_label}', fontsize=14, fontweight='bold',
                 y=1.02)
    plt.tight_layout()
    fig.savefig(str(fig_path) + '.pdf', format='pdf')
    fig.savefig(str(fig_path) + '.png')
    plt.close(fig)


def add_log_variants(normalized, pseudocount=1e-5):
    """Create log-transformed versions of each normalization method.

    CLR is skipped since it already applies a log transform.
    Returns a dict with keys like 'log_none', 'log_total_sum', etc.
    """
    log_variants = {}
    for method_name, X_norm in normalized.items():
        if method_name == 'clr':
            continue
        X_pos = np.maximum(X_norm, 0) + pseudocount
        log_variants[f'log_{method_name}'] = np.log(X_pos)
    return log_variants


def test_analyte_normality(X, analyte_names, alpha=0.05):
    """Test each analyte for normality using Shapiro-Wilk and D'Agostino-Pearson.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_subjects × n_analytes).
    analyte_names : list
        Names of analytes (columns of X).
    alpha : float
        Significance level for rejection.

    Returns
    -------
    pd.DataFrame with columns: analyte, shapiro_stat, shapiro_p,
        dagostino_stat, dagostino_p, classification
        ('normal' if both tests fail to reject at alpha, 'non-normal' otherwise).
    """
    rows = []
    for j, name in enumerate(analyte_names):
        col = X[:, j]
        col_clean = col[~np.isnan(col)]

        shapiro_stat = shapiro_p = dagostino_stat = dagostino_p = np.nan
        classification = 'normal'

        if len(col_clean) >= 8:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(col_clean)
            except Exception:
                pass

            try:
                dagostino_stat, dagostino_p = stats.normaltest(col_clean)
            except Exception:
                pass

            # Classify: non-normal if either test rejects
            reject_shapiro = (shapiro_p is not None
                              and not np.isnan(shapiro_p)
                              and shapiro_p < alpha)
            reject_dagostino = (dagostino_p is not None
                                and not np.isnan(dagostino_p)
                                and dagostino_p < alpha)
            if reject_shapiro or reject_dagostino:
                classification = 'non-normal'

        rows.append({
            'analyte': name,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'dagostino_stat': dagostino_stat,
            'dagostino_p': dagostino_p,
            'classification': classification,
        })

    return pd.DataFrame(rows)


def plot_log_comparison(normalized, log_variants, y, analyte_names,
                        fig_path, title_label, n_permutations=999):
    """Analyze and plot the effect of log-transforming each normalization.

    Creates a 1x3 figure:
      A) # significant analytes: raw vs log for each method, CLR as reference
      B) PERMANOVA R² (Euclidean): raw vs log, CLR as reference
      C) Scatter: -log10(p) simple log vs CLR

    Returns comparison DataFrame saved as CSV.
    """
    base_methods = [m for m in normalized if m != 'clr']
    rows = []

    # --- CLR reference ---
    res_clr = differential_analysis(normalized['clr'], y)
    clr_n_sig = int((res_clr['p_value'] < 0.05).sum())

    valid_clr = ~np.isnan(normalized['clr']).any(axis=1)
    D_clr = compute_distance_matrix(normalized['clr'][valid_clr], 'euclidean')
    clr_perm = permanova_r2(D_clr, y[valid_clr], n_permutations=n_permutations)
    clr_r2 = clr_perm['r2']

    rows.append({'method': 'clr', 'scale': 'log (CLR)',
                 'n_sig_nominal': clr_n_sig, 'permanova_r2': clr_r2})

    # --- Each base method: raw + log ---
    for method in base_methods:
        for scale, X_data in [('raw', normalized[method]),
                              ('log', log_variants[f'log_{method}'])]:
            res = differential_analysis(X_data, y)
            n_sig = int((res['p_value'] < 0.05).sum())

            valid = ~np.isnan(X_data).any(axis=1)
            try:
                D = compute_distance_matrix(X_data[valid], 'euclidean')
                perm = permanova_r2(D, y[valid], n_permutations=n_permutations)
                r2 = perm['r2']
            except Exception:
                r2 = np.nan

            rows.append({'method': method, 'scale': scale,
                         'n_sig_nominal': n_sig, 'permanova_r2': r2})

    comparison_df = pd.DataFrame(rows)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(base_methods))
    width = 0.35

    raw_sigs = [comparison_df[(comparison_df['method'] == m) &
                (comparison_df['scale'] == 'raw')]['n_sig_nominal'].values[0]
                for m in base_methods]
    log_sigs = [comparison_df[(comparison_df['method'] == m) &
                (comparison_df['scale'] == 'log')]['n_sig_nominal'].values[0]
                for m in base_methods]

    # Panel A: # significant analytes
    ax = axes[0]
    ax.bar(x - width/2, raw_sigs, width, label='Raw scale',
           color='#E69F00', alpha=0.8)
    ax.bar(x + width/2, log_sigs, width, label='Log scale',
           color='#56B4E9', alpha=0.8)
    ax.axhline(clr_n_sig, color='#009E73', linestyle='--', linewidth=2,
               label=f'CLR ({clr_n_sig})')
    ax.set_xticks(x)
    ax.set_xticklabels(base_methods, rotation=45, ha='right')
    ax.set_ylabel('# Significant Analytes (p < 0.05)')
    ax.set_title('A) Significance: Raw vs Log Scale', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel B: PERMANOVA R² (Euclidean)
    ax = axes[1]
    raw_r2s = [comparison_df[(comparison_df['method'] == m) &
               (comparison_df['scale'] == 'raw')]['permanova_r2'].values[0]
               for m in base_methods]
    log_r2s = [comparison_df[(comparison_df['method'] == m) &
               (comparison_df['scale'] == 'log')]['permanova_r2'].values[0]
               for m in base_methods]

    ax.bar(x - width/2, raw_r2s, width, label='Raw scale',
           color='#E69F00', alpha=0.8)
    ax.bar(x + width/2, log_r2s, width, label='Log scale',
           color='#56B4E9', alpha=0.8)
    ax.axhline(clr_r2, color='#009E73', linestyle='--', linewidth=2,
               label=f'CLR (R²={clr_r2:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(base_methods, rotation=45, ha='right')
    ax.set_ylabel('PERMANOVA R² (Euclidean)')
    ax.set_title('B) Group Separation: Raw vs Log Scale', fontweight='bold')
    ax.legend(fontsize=8)

    # Panel C: Scatter — simple log vs CLR p-values
    ax = axes[2]
    res_log_none = differential_analysis(log_variants['log_none'], y)
    p_log = np.clip(res_log_none['p_value'].values, 1e-300, None)
    p_clr = np.clip(res_clr['p_value'].values, 1e-300, None)
    x_s = -np.log10(p_log)
    y_s = -np.log10(p_clr)
    valid_s = ~(np.isnan(x_s) | np.isnan(y_s))

    ax.scatter(x_s[valid_s], y_s[valid_s], c='#CC79A7', alpha=0.7, s=40,
               edgecolors='none')
    max_v = max(x_s[valid_s].max(), y_s[valid_s].max()) * 1.1
    ax.plot([0, max_v], [0, max_v], 'k--', alpha=0.4)
    ax.axhline(-np.log10(0.05), color='gray', linestyle=':', alpha=0.3)
    ax.axvline(-np.log10(0.05), color='gray', linestyle=':', alpha=0.3)

    r_val, _ = stats.pearsonr(x_s[valid_s], y_s[valid_s])
    ax.set_xlabel(r'$-\log_{10}(p)$ Simple Log')
    ax.set_ylabel(r'$-\log_{10}(p)$ CLR')
    ax.set_title(f'C) Simple Log vs CLR (r = {r_val:.2f})', fontweight='bold')

    fig.suptitle(f'Figure {title_label}', fontsize=14, fontweight='bold',
                 y=1.02)
    plt.tight_layout()
    fig.savefig(str(fig_path) + '.pdf', format='pdf')
    fig.savefig(str(fig_path) + '.png')
    plt.close(fig)

    # Print summary
    print(f"  CLR: {clr_n_sig} sig, R²={clr_r2:.3f}")
    for method in base_methods:
        r_raw = comparison_df[(comparison_df['method'] == method) &
                (comparison_df['scale'] == 'raw')]
        r_log = comparison_df[(comparison_df['method'] == method) &
                (comparison_df['scale'] == 'log')]
        print(f"  {method}: raw {r_raw['n_sig_nominal'].values[0]} sig / "
              f"R²={r_raw['permanova_r2'].values[0]:.3f} → "
              f"log {r_log['n_sig_nominal'].values[0]} sig / "
              f"R²={r_log['permanova_r2'].values[0]:.3f}")
    print(f"  Simple log vs CLR p-value correlation: r={r_val:.3f}")

    return comparison_df


def plot_log_comparison_stratified(normalized, log_variants, y, analyte_names,
                                    normality_df, fig_path, title_label,
                                    n_permutations=999):
    """Stratified log-comparison by analyte distribution type.

    Splits analytes into normal vs non-normal (based on normality_df) and
    repeats differential analysis + PERMANOVA for each stratum separately.

    Creates a 2×2 figure:
      Top row: Non-normal analytes — # significant + PERMANOVA R²
      Bottom row: Normal analytes — # significant + PERMANOVA R²

    Returns stratified comparison DataFrame.
    """
    base_methods = [m for m in normalized if m != 'clr']
    strata = {'non-normal': [], 'normal': []}

    for _, row in normality_df.iterrows():
        idx = analyte_names.index(row['analyte']) if row['analyte'] in analyte_names else -1
        if idx >= 0:
            strata[row['classification']].append(idx)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    all_rows = []

    for row_idx, (stratum_name, analyte_idx) in enumerate(strata.items()):
        n_analytes = len(analyte_idx)

        if n_analytes == 0:
            for col_idx in range(2):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, f'No {stratum_name}\nanalytes',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=14)
                ax.set_title(f'{stratum_name.title()} Analytes (N=0)',
                             fontweight='bold')
            continue

        # Subset data matrices to these analyte columns
        X_clr_sub = normalized['clr'][:, analyte_idx]
        res_clr = differential_analysis(X_clr_sub, y)
        clr_n_sig = int((res_clr['p_value'] < 0.05).sum())

        valid_clr = ~np.isnan(X_clr_sub).any(axis=1)
        try:
            D_clr = compute_distance_matrix(X_clr_sub[valid_clr], 'euclidean')
            clr_perm = permanova_r2(D_clr, y[valid_clr],
                                     n_permutations=n_permutations)
            clr_r2 = clr_perm['r2']
        except Exception:
            clr_r2 = np.nan

        raw_sigs = []
        log_sigs = []
        raw_r2s = []
        log_r2s = []

        for method in base_methods:
            X_raw_sub = normalized[method][:, analyte_idx]
            X_log_sub = log_variants[f'log_{method}'][:, analyte_idx]

            res_raw = differential_analysis(X_raw_sub, y)
            res_log = differential_analysis(X_log_sub, y)
            raw_sigs.append(int((res_raw['p_value'] < 0.05).sum()))
            log_sigs.append(int((res_log['p_value'] < 0.05).sum()))

            for scale, X_sub in [('raw', X_raw_sub), ('log', X_log_sub)]:
                valid = ~np.isnan(X_sub).any(axis=1)
                try:
                    D = compute_distance_matrix(X_sub[valid], 'euclidean')
                    perm = permanova_r2(D, y[valid],
                                         n_permutations=n_permutations)
                    r2 = perm['r2']
                except Exception:
                    r2 = np.nan
                if scale == 'raw':
                    raw_r2s.append(r2)
                else:
                    log_r2s.append(r2)

                all_rows.append({
                    'stratum': stratum_name,
                    'n_analytes': n_analytes,
                    'method': method,
                    'scale': scale,
                    'n_sig': int(((res_raw if scale == 'raw' else res_log)['p_value'] < 0.05).sum()),
                    'permanova_r2': r2,
                })

        all_rows.append({
            'stratum': stratum_name,
            'n_analytes': n_analytes,
            'method': 'clr',
            'scale': 'log (CLR)',
            'n_sig': clr_n_sig,
            'permanova_r2': clr_r2,
        })

        # Panel A: # significant analytes
        ax = axes[row_idx, 0]
        x = np.arange(len(base_methods))
        width = 0.35
        ax.bar(x - width / 2, raw_sigs, width, label='Raw scale',
               color='#E69F00', alpha=0.8)
        ax.bar(x + width / 2, log_sigs, width, label='Log scale',
               color='#56B4E9', alpha=0.8)
        ax.axhline(clr_n_sig, color='#009E73', linestyle='--', linewidth=2,
                    label=f'CLR ({clr_n_sig})')
        ax.set_xticks(x)
        ax.set_xticklabels(base_methods, rotation=45, ha='right')
        ax.set_ylabel('# Significant (p < 0.05)')
        ax.set_title(f'{stratum_name.title()} Analytes (N={n_analytes}) — '
                     f'Significance', fontweight='bold')
        ax.legend(fontsize=7)

        # Panel B: PERMANOVA R²
        ax = axes[row_idx, 1]
        ax.bar(x - width / 2, raw_r2s, width, label='Raw scale',
               color='#E69F00', alpha=0.8)
        ax.bar(x + width / 2, log_r2s, width, label='Log scale',
               color='#56B4E9', alpha=0.8)
        ax.axhline(clr_r2, color='#009E73', linestyle='--', linewidth=2,
                    label=f'CLR (R²={clr_r2:.3f})' if not np.isnan(clr_r2) else 'CLR')
        ax.set_xticks(x)
        ax.set_xticklabels(base_methods, rotation=45, ha='right')
        ax.set_ylabel('PERMANOVA R² (Euclidean)')
        ax.set_title(f'{stratum_name.title()} Analytes (N={n_analytes}) — '
                     f'Group Separation', fontweight='bold')
        ax.legend(fontsize=7)

    fig.suptitle(f'Figure {title_label}', fontsize=14, fontweight='bold',
                 y=1.02)
    plt.tight_layout()
    fig.savefig(str(fig_path) + '.pdf', format='pdf')
    fig.savefig(str(fig_path) + '.png')
    plt.close(fig)

    strat_df = pd.DataFrame(all_rows)
    # Save CSV in results directory
    csv_path = str(fig_path).replace('figures/', 'results/') + '.csv'
    strat_df.to_csv(csv_path, index=False)

    # Print summary
    for stratum_name in ['non-normal', 'normal']:
        sub = strat_df[strat_df['stratum'] == stratum_name]
        if len(sub) > 0:
            n_a = sub['n_analytes'].iloc[0]
            print(f"  {stratum_name} (N={n_a}): CLR sig={sub[sub['method']=='clr']['n_sig'].values[0] if len(sub[sub['method']=='clr']) else 'N/A'}")

    return strat_df


def pca_centroid_comparison_realdata(normalized, y, methods_ordered,
                                     n_components=2, n_permutations=999):
    """Run PCA centroid analysis across normalizations × preprocessing configs.

    Parameters
    ----------
    normalized : dict
        {method_name: X_normalized} from apply_normalizations().
    y : np.ndarray
        Binary group labels.
    methods_ordered : list
        Normalization method names to iterate over.
    n_components : int
        Number of PCA components.
    n_permutations : int
        Number of permutations for all tests.

    Returns
    -------
    pd.DataFrame with one row per norm_method × preprocessing combination.
    """
    preproc_configs = [
        ('raw',        False, False),
        ('log',        True,  False),
        ('scaled',     False, True),
        ('log+scaled', True,  True),
    ]

    rows = []
    for method_name in methods_ordered:
        X_norm = normalized[method_name]
        for preproc_label, do_log, do_scale in preproc_configs:
            X_prep = prepare_pca_data(X_norm, method_name,
                                      log_transform=do_log, scale=do_scale)
            result = pca_centroid_analysis(
                X_prep, y, n_components=n_components,
                n_permutations=n_permutations, scale=False)
            result['norm_method'] = method_name
            result['preprocessing'] = preproc_label
            rows.append(result)

    return pd.DataFrame(rows)


def plot_pca_centroid_heatmap(centroid_df, fig_path, title_label):
    """1×3 heatmap: one panel per test method, colored by -log10(p).

    Parameters
    ----------
    centroid_df : pd.DataFrame
        Output of pca_centroid_comparison_realdata().
    fig_path : str or Path
        Output path (without extension).
    title_label : str
        Title prefix for the figure.
    """
    test_specs = [
        ('hotelling_permutation_p', "Hotelling's T² (perm)"),
        ('permanova_p',             'PERMANOVA'),
        ('centroid_distance_p',     'Centroid Distance'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for col_idx, (pcol, tlab) in enumerate(test_specs):
        ax = axes[col_idx]
        df_plot = centroid_df.copy()
        df_plot['neglog10p'] = -np.log10(df_plot[pcol].clip(lower=1e-10))

        pivot = df_plot.pivot_table(
            index='norm_method', columns='preprocessing',
            values='neglog10p', aggfunc='first')

        # Reorder
        norm_order = [m for m in ['none', 'total_sum', 'pqn', 'clr',
                                   'median', 'quantile', 'protein_corrected']
                      if m in pivot.index]
        preproc_order = [p for p in ['raw', 'log', 'scaled', 'log+scaled']
                         if p in pivot.columns]
        pivot = pivot.reindex(index=norm_order, columns=preproc_order)

        # Annotation: value + significance star
        annot_matrix = pivot.copy().astype(str)
        p_pivot = df_plot.pivot_table(
            index='norm_method', columns='preprocessing',
            values=pcol, aggfunc='first')
        p_pivot = p_pivot.reindex(index=norm_order, columns=preproc_order)
        for i in range(len(norm_order)):
            for j in range(len(preproc_order)):
                val = pivot.iloc[i, j]
                pval = p_pivot.iloc[i, j]
                if np.isnan(val):
                    annot_matrix.iloc[i, j] = ''
                else:
                    star = '***' if pval < 0.001 else ('**' if pval < 0.01
                           else ('*' if pval < 0.05 else ''))
                    annot_matrix.iloc[i, j] = f'{val:.1f}{star}'

        sns.heatmap(pivot, annot=annot_matrix, fmt='', cmap='viridis',
                    ax=ax, linewidths=0.5,
                    cbar_kws={'label': '$-\\log_{10}(p)$'})
        sig_line = -np.log10(0.05)
        ax.set_title(f'{tlab}\n(dashed = p=0.05: {sig_line:.1f})',
                     fontweight='bold', fontsize=10)
        if col_idx > 0:
            ax.set_ylabel('')

    fig.suptitle(f'Figure {title_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(fig_path) + '.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(str(fig_path) + '.png', bbox_inches='tight')
    plt.close(fig)


def plot_pca_with_centroids_and_ellipses(normalized, y, methods_ordered,
                                          fig_path, title_label,
                                          group_labels=('Group 0', 'Group 1'),
                                          group_colors=('#0072B2', '#D55E00')):
    """2×N PCA scatter grid with group centroids and 95% confidence ellipses.

    Row 0: raw (no log, no scale).  Row 1: log+scaled.
    Columns: normalization methods.

    Parameters
    ----------
    normalized : dict
        {method_name: X_normalized}.
    y : np.ndarray
        Binary group labels.
    methods_ordered : list
        Normalization methods.
    fig_path : str or Path
        Output path without extension.
    title_label : str
        Figure title prefix.
    group_labels : tuple
        Display names for the two groups.
    group_colors : tuple
        Colors for the two groups.
    """
    from matplotlib.patches import Ellipse

    n_methods = len(methods_ordered)
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    preproc_configs = [
        ('Raw',        False, False),
        ('Log+Scaled', True,  True),
    ]

    for row_idx, (preproc_label, do_log, do_scale) in enumerate(preproc_configs):
        for col_idx, method_name in enumerate(methods_ordered):
            ax = axes[row_idx, col_idx]
            X_prep = prepare_pca_data(normalized[method_name], method_name,
                                      log_transform=do_log, scale=do_scale)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_prep)

            groups = np.unique(y)
            for g, label, color in zip(groups, group_labels, group_colors):
                mask = y == g
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color,
                           alpha=0.6, s=40, edgecolors='none', label=label)

                # Centroid
                centroid = X_pca[mask].mean(axis=0)
                ax.scatter(centroid[0], centroid[1], c=color, marker='X',
                           s=150, edgecolors='black', linewidths=1.5,
                           zorder=5)

                # 95% confidence ellipse via eigendecomposition
                if mask.sum() >= 3:
                    cov_matrix = np.cov(X_pca[mask], rowvar=False)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    # Sort descending
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]

                    # Chi-squared scaling for 95% confidence (2 DOF)
                    from scipy.stats import chi2
                    chi2_val = chi2.ppf(0.95, 2)
                    width = 2 * np.sqrt(chi2_val * max(eigenvalues[0], 0))
                    height = 2 * np.sqrt(chi2_val * max(eigenvalues[1], 0))

                    angle = np.degrees(np.arctan2(eigenvectors[1, 0],
                                                   eigenvectors[0, 0]))
                    ellipse = Ellipse(xy=centroid, width=width, height=height,
                                      angle=angle, facecolor=color,
                                      alpha=0.15, edgecolor=color,
                                      linewidth=1.5, linestyle='--')
                    ax.add_patch(ellipse)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title(f'{method_name}: {preproc_label}',
                         fontweight='bold', fontsize=10)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f'Figure {title_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(fig_path) + '.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(str(fig_path) + '.png', bbox_inches='tight')
    plt.close(fig)


def plot_pca_centroid_heatmap_comparison(balf_df, blood_df, fig_path,
                                          title_label):
    """Side-by-side BALF vs Blood PCA centroid heatmap comparison.

    Parameters
    ----------
    balf_df : pd.DataFrame
        PCA centroid results for BALF.
    blood_df : pd.DataFrame
        PCA centroid results for Blood.
    fig_path : str or Path
        Output path without extension.
    title_label : str
        Figure title prefix.
    """
    test_specs = [
        ('hotelling_permutation_p', "Hotelling's T²"),
        ('permanova_p',             'PERMANOVA'),
        ('centroid_distance_p',     'Centroid Distance'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for row_idx, (data_df, tissue_label) in enumerate(
            [(balf_df, 'BALF'), (blood_df, 'Blood')]):
        for col_idx, (pcol, tlab) in enumerate(test_specs):
            ax = axes[row_idx, col_idx]
            df_plot = data_df.copy()
            df_plot['neglog10p'] = -np.log10(df_plot[pcol].clip(lower=1e-10))

            pivot = df_plot.pivot_table(
                index='norm_method', columns='preprocessing',
                values='neglog10p', aggfunc='first')

            norm_order = [m for m in ['none', 'total_sum', 'pqn', 'clr',
                                       'median', 'quantile']
                          if m in pivot.index]
            preproc_order = [p for p in ['raw', 'log', 'scaled', 'log+scaled']
                             if p in pivot.columns]
            pivot = pivot.reindex(index=norm_order, columns=preproc_order)

            # Annotation with stars
            annot_matrix = pivot.copy().astype(str)
            p_pivot = df_plot.pivot_table(
                index='norm_method', columns='preprocessing',
                values=pcol, aggfunc='first')
            p_pivot = p_pivot.reindex(index=norm_order, columns=preproc_order)
            for i in range(len(norm_order)):
                for j in range(len(preproc_order)):
                    val = pivot.iloc[i, j]
                    pval = p_pivot.iloc[i, j]
                    if np.isnan(val):
                        annot_matrix.iloc[i, j] = ''
                    else:
                        star = '***' if pval < 0.001 else ('**' if pval < 0.01
                               else ('*' if pval < 0.05 else ''))
                        annot_matrix.iloc[i, j] = f'{val:.1f}{star}'

            sns.heatmap(pivot, annot=annot_matrix, fmt='', cmap='viridis',
                        ax=ax, linewidths=0.5,
                        cbar_kws={'label': '$-\\log_{10}(p)$'})
            ax.set_title(f'{tissue_label}: {tlab}',
                         fontweight='bold', fontsize=10)
            if col_idx > 0:
                ax.set_ylabel('')

    fig.suptitle(f'Figure {title_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(str(fig_path) + '.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(str(fig_path) + '.png', bbox_inches='tight')
    plt.close(fig)


def run_picflu_reanalysis():
    """Main reanalysis pipeline."""
    print("=" * 60)
    print("PICFLU COHORT REANALYSIS")
    print("Boyd et al. 2020, Nature (PMID 33116313)")
    print("=" * 60)

    # Load data
    df, analyte_cols = load_picflu_data()

    # Prepare data matrix (complete cases for the selected analytes)
    X_raw = df[analyte_cols].values
    y = df['outcome'].values

    # Strategy: use per-analyte analysis (not requiring all-complete rows)
    # Select analytes with >= 40 non-missing values
    missing_pct = np.isnan(X_raw).mean(axis=0)
    core_analytes = [analyte_cols[i] for i in range(len(analyte_cols))
                     if missing_pct[i] < 0.5]
    print(f"\nCore analytes (< 50% missing): {len(core_analytes)}")

    # For normalization: use a subset of analytes with the best coverage
    # to build the normalization reference, then analyze all core analytes
    # Use subjects with valid outcome
    valid_subjects = ~np.isnan(y)
    X_all = df.loc[valid_subjects, core_analytes].values
    y_valid = y[valid_subjects]
    total_protein = df.loc[valid_subjects, 'Total Protein'].values

    # For CLR/PQN/etc, we need a matrix without NaN. Use the subset of
    # analytes with <= 40% missing for normalization computation.
    norm_analyte_mask = np.isnan(X_all).mean(axis=0) < 0.4
    norm_analyte_idx = np.where(norm_analyte_mask)[0]
    norm_analytes = [core_analytes[i] for i in norm_analyte_idx]
    X_norm_base = X_all[:, norm_analyte_mask]

    # Impute remaining NaN in norm_base with column medians for normalization
    for j in range(X_norm_base.shape[1]):
        col = X_norm_base[:, j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med
        X_norm_base[:, j] = col

    # Ensure positivity for CLR
    X_norm_base[X_norm_base <= 0] = np.nanmin(X_norm_base[X_norm_base > 0]) / 2

    n_subjects = X_norm_base.shape[0]
    print(f"Normalization base: {n_subjects} subjects, {len(norm_analytes)} analytes")
    print(f"  Group 0 (no PrAHRF/Death): {(y_valid == 0).sum()}")
    print(f"  Group 1 (PrAHRF/Death): {(y_valid == 1).sum()}")

    # Apply normalizations to the normalization base
    print("\nApplying normalization methods...")
    normalized = apply_normalizations(X_norm_base, norm_analytes)

    # Differential analysis under each normalization
    print("\nPerforming differential analysis...")
    all_results = {}
    sig_sets = {}

    for method_name, X_norm in normalized.items():
        res_df = differential_analysis(X_norm, y_valid)
        res_df['analyte'] = norm_analytes
        res_df['method'] = method_name

        # FDR correction
        valid_mask = ~res_df['p_value'].isna()
        if valid_mask.any():
            p_valid = res_df.loc[valid_mask, 'p_value'].values
            p_fdr, sig_fdr = multiple_testing_correction(p_valid, 'fdr_bh')
            res_df.loc[valid_mask, 'p_fdr'] = p_fdr
            res_df.loc[valid_mask, 'sig_fdr'] = sig_fdr

            # Also nominal significance
            res_df['sig_nominal'] = res_df['p_value'] < 0.05
        else:
            res_df['p_fdr'] = np.nan
            res_df['sig_fdr'] = False
            res_df['sig_nominal'] = False

        all_results[method_name] = res_df
        sig_sets[method_name] = set(
            res_df.loc[res_df['sig_nominal'] == True, 'analyte'].tolist()
        )

        n_sig_nominal = res_df['sig_nominal'].sum()
        n_sig_fdr = res_df['sig_fdr'].sum() if 'sig_fdr' in res_df else 0
        print(f"  {method_name}: {n_sig_nominal} significant (nominal), "
              f"{n_sig_fdr} significant (FDR)")

    # Update core_analytes to norm_analytes for plotting
    core_analytes = norm_analytes
    X_complete = X_norm_base
    y_complete = y_valid
    tp_complete = total_protein

    # Per-analyte normality testing
    print("\n--- Per-Analyte Normality Testing ---")
    normality_df = test_analyte_normality(X_norm_base, core_analytes)
    normality_df.to_csv(RESULTS_DIR / 'picflu_normality_tests.csv', index=False)
    n_normal = (normality_df['classification'] == 'normal').sum()
    n_nonnormal = (normality_df['classification'] == 'non-normal').sum()
    print(f"  Normal: {n_normal}, Non-normal: {n_nonnormal}")

    # Combine all results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_df.to_csv(RESULTS_DIR / 'picflu_reanalysis_results.csv', index=False)

    # --- Figure E4: PICFLU Reanalysis ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Number of significant analytes by method
    ax = axes[0]
    methods_ordered = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile']
    n_sig_nominal = [combined_df[(combined_df['method'] == m) & (combined_df['sig_nominal'] == True)].shape[0]
                     for m in methods_ordered]
    n_sig_fdr = [combined_df[(combined_df['method'] == m) & (combined_df['sig_fdr'] == True)].shape[0]
                 for m in methods_ordered]

    x = np.arange(len(methods_ordered))
    width = 0.35
    ax.bar(x - width / 2, n_sig_nominal, width, label='Nominal (p < 0.05)',
           color='#D55E00', alpha=0.8)
    ax.bar(x + width / 2, n_sig_fdr, width, label='FDR-corrected',
           color='#0072B2', alpha=0.8)
    ax.set_xlabel('Normalization Method')
    ax.set_ylabel('Number of Significant Analytes')
    ax.set_title('A) Significant Analytes by Method', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_ordered, rotation=45, ha='right')
    ax.legend(fontsize=9)

    # Panel B: Heatmap of significance concordance
    ax = axes[1]
    sig_matrix = pd.DataFrame(index=core_analytes, columns=methods_ordered, data=False)
    for method in methods_ordered:
        res = all_results[method]
        for _, row in res.iterrows():
            if row['sig_nominal']:
                sig_matrix.loc[row['analyte'], method] = True

    sig_numeric = sig_matrix.astype(int)
    # Only show analytes significant in at least one method
    any_sig = sig_numeric.sum(axis=1) > 0
    if any_sig.any():
        sig_show = sig_numeric.loc[any_sig]
        sns.heatmap(sig_show, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Significant'},
                    linewidths=0.5, linecolor='gray')
        ax.set_title('B) Significance Concordance (p < 0.05)', fontweight='bold')
        ax.set_xlabel('Normalization Method')
        ax.set_ylabel('Analyte')
    else:
        ax.text(0.5, 0.5, 'No significant\nanalytes found', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('B) Significance Concordance', fontweight='bold')

    # Panel C: Pairwise Jaccard similarity
    ax = axes[2]
    jaccard_matrix = np.zeros((len(methods_ordered), len(methods_ordered)))
    for i, m1 in enumerate(methods_ordered):
        for j, m2 in enumerate(methods_ordered):
            s1 = sig_sets.get(m1, set())
            s2 = sig_sets.get(m2, set())
            union = s1 | s2
            if len(union) > 0:
                jaccard_matrix[i, j] = len(s1 & s2) / len(union)
            else:
                jaccard_matrix[i, j] = 1.0 if i == j else 0.0

    jaccard_df = pd.DataFrame(jaccard_matrix, index=methods_ordered, columns=methods_ordered)
    sns.heatmap(jaccard_df, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                vmin=0, vmax=1, linewidths=0.5, square=True)
    ax.set_title('C) Pairwise Jaccard Similarity', fontweight='bold')

    fig.suptitle('Figure E4: PICFLU Cohort BAL Data Reanalysis',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE4_picflu_reanalysis.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE4_picflu_reanalysis.png')
    plt.close(fig)

    # --- Figure E9: Volcano Plots ---
    methods_ordered = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, method in enumerate(methods_ordered):
        ax = axes[idx // 3, idx % 3]
        res = all_results[method].copy()
        res = res.dropna(subset=['p_value', 'fold_change'])

        # Compute log2 fold change and -log10 p-value
        res['log2fc'] = np.log2(res['fold_change'].clip(lower=1e-10))
        res['neg_log10p'] = -np.log10(res['p_value'].clip(lower=1e-300))

        # Color by significance
        sig_mask = res['sig_nominal'] == True
        ax.scatter(res.loc[~sig_mask, 'log2fc'], res.loc[~sig_mask, 'neg_log10p'],
                   c='#999999', alpha=0.6, s=20, edgecolors='none', label='NS')
        ax.scatter(res.loc[sig_mask, 'log2fc'], res.loc[sig_mask, 'neg_log10p'],
                   c='#D55E00', alpha=0.8, s=30, edgecolors='none', label='p < 0.05')

        # Label top 5 by p-value
        top5 = res.nsmallest(5, 'p_value')
        for _, row in top5.iterrows():
            label = row['analyte'] if len(str(row['analyte'])) < 15 else str(row['analyte'])[:12] + '...'
            ax.annotate(label, (row['log2fc'], row['neg_log10p']),
                        fontsize=6, ha='center', va='bottom')

        ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('log₂(Fold Change)')
        ax.set_ylabel('-log₁₀(p-value)')
        ax.set_title(f'{method}', fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Figure E9: Volcano Plots by Normalization Method (PICFLU)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE9_picflu_volcanos.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE9_picflu_volcanos.png')
    plt.close(fig)

    # --- Figure E10: PCA by Normalization (log-transformed + scaled) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, method in enumerate(methods_ordered):
        ax = axes[idx // 3, idx % 3]
        X_prep = prepare_pca_data(normalized[method], method,
                                  log_transform=True, scale=True)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_prep)

        for g, label, color in [(0, 'Control', '#0072B2'), (1, 'PrAHRF/Death', '#D55E00')]:
            mask = y_complete == g
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.6, s=25,
                       edgecolors='none', label=label)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        preproc = 'scaled' if method == 'clr' else 'log + scaled'
        ax.set_title(f'{method} ({preproc})', fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Figure E10: PCA Biplots by Normalization Method (PICFLU)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE10_picflu_pca.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE10_picflu_pca.png')
    plt.close(fig)

    # --- Figure E10b: PCA Preprocessing Comparison (PICFLU) ---
    preproc_configs = [
        ('Raw', False, False),
        ('Log-transformed', True, False),
        ('Centered + Scaled', False, True),
        ('Log + Centered + Scaled', True, True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    # Row 0: "none" normalization; Row 1: "clr" normalization
    for row, (norm_method, norm_label) in enumerate([('none', 'No Normalization'),
                                                      ('clr', 'CLR')]):
        for col, (preproc_label, do_log, do_scale) in enumerate(preproc_configs):
            ax = axes[row, col]
            X_prep = prepare_pca_data(normalized[norm_method], norm_method,
                                      log_transform=do_log, scale=do_scale)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_prep)

            for g, label, color in [(0, 'Control', '#0072B2'),
                                     (1, 'PrAHRF/Death', '#D55E00')]:
                mask = y_complete == g
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.6,
                           s=25, edgecolors='none', label=label)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title(f'{norm_label}: {preproc_label}', fontweight='bold',
                         fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    fig.suptitle('Figure E10b: PCA Preprocessing Comparison (PICFLU)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE10b_picflu_pca_preprocessing.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE10b_picflu_pca_preprocessing.png')
    plt.close(fig)

    # --- Figure E11: Analyte Rank Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Build rank matrix: rank analytes by |test_stat| under each method
    rank_matrix = pd.DataFrame(index=norm_analytes)
    for method in methods_ordered:
        res = all_results[method].copy()
        res = res.set_index('analyte')
        rank_matrix[method] = res['test_stat'].abs().rank(ascending=False)

    # Panel A: Spearman correlation of analyte rankings across methods
    ax = axes[0]
    rank_corr = rank_matrix.corr(method='spearman')
    sns.heatmap(rank_corr, ax=ax, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1, linewidths=0.5, square=True)
    ax.set_title('A) Rank Correlation Across Methods', fontweight='bold')

    # Panel B: Rank of top-10 analytes (from raw analysis) across normalizations
    ax = axes[1]
    raw_top10 = rank_matrix['none'].nsmallest(10).index.tolist()
    rank_top10 = rank_matrix.loc[raw_top10, methods_ordered]
    # Shorten analyte names for display
    short_names = [n[:18] if len(n) > 18 else n for n in rank_top10.index]
    rank_display = rank_top10.copy()
    rank_display.index = short_names
    sns.heatmap(rank_display, ax=ax, annot=True, fmt='.0f', cmap='YlOrRd_r',
                linewidths=0.5, cbar_kws={'label': 'Rank'})
    ax.set_title('B) Top-10 Analyte Ranks Across Methods', fontweight='bold')
    ax.set_ylabel('Analyte (ranked by raw analysis)')
    ax.set_xlabel('Normalization Method')

    fig.suptitle('Figure E11: Analyte Ranking Comparison Across Normalization Methods',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE11_picflu_ranks.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE11_picflu_ranks.png')
    plt.close(fig)

    print(f"  Saved: figE9_picflu_volcanos.pdf, figE10_picflu_pca.pdf, figE10b_picflu_pca_preprocessing.pdf, figE11_picflu_ranks.pdf")

    # --- Distance metric analysis ---
    print("\n--- Distance Metric Analysis (PERMANOVA R²) ---")
    picflu_distance_df = distance_metric_analysis(
        normalized, y_complete, methods_ordered, n_permutations=999)
    picflu_distance_df.to_csv(RESULTS_DIR / 'picflu_distance_results.csv',
                              index=False)
    print(f"  Computed {len(picflu_distance_df)} norm×metric combinations")

    plot_permanova_heatmap(
        picflu_distance_df,
        FIGURES_DIR / 'figE18_picflu_distance',
        'E18: PICFLU Distance Metric Analysis')
    print("  Saved: figE18_picflu_distance.pdf")

    # Best distance metric summary
    if len(picflu_distance_df) > 0:
        best_row = picflu_distance_df.loc[picflu_distance_df['permanova_r2'].idxmax()]
        print(f"  Best: {best_row['norm_method']}/{best_row['distance_metric']} "
              f"R²={best_row['permanova_r2']:.3f} (p={best_row['permanova_p']:.3f})")

    # --- Log transform comparison ---
    print("\n--- Log Transform Comparison ---")
    log_variants = add_log_variants(normalized)
    picflu_log_df = plot_log_comparison(
        normalized, log_variants, y_complete, core_analytes,
        FIGURES_DIR / 'figE20_picflu_log_comparison',
        'E20: PICFLU Log Transform Comparison', n_permutations=999)
    picflu_log_df.to_csv(RESULTS_DIR / 'picflu_log_comparison.csv', index=False)
    print("  Saved: figE20_picflu_log_comparison.pdf, picflu_log_comparison.csv")

    # --- Stratified log comparison (by analyte distribution) ---
    print("\n--- Stratified Log Comparison (by Analyte Distribution) ---")
    picflu_log_strat_df = plot_log_comparison_stratified(
        normalized, log_variants, y_complete, core_analytes, normality_df,
        FIGURES_DIR / 'figE20b_picflu_log_stratified',
        'E20b: PICFLU Log Comparison (by Analyte Distribution)')
    print("  Saved: figE20b_picflu_log_stratified.pdf")

    # --- PCA Centroid Comparison ---
    print("\n--- PCA Centroid Comparison ---")
    picflu_centroid_df = pca_centroid_comparison_realdata(
        normalized, y_complete, methods_ordered,
        n_components=2, n_permutations=999)
    picflu_centroid_df.to_csv(
        RESULTS_DIR / 'picflu_pca_centroid_results.csv', index=False)
    print(f"  Computed {len(picflu_centroid_df)} norm×preprocessing combinations")

    plot_pca_centroid_heatmap(
        picflu_centroid_df,
        FIGURES_DIR / 'figE26_picflu_pca_centroid',
        'E26: PICFLU PCA Centroid Comparison')
    print("  Saved: figE26_picflu_pca_centroid.pdf")

    plot_pca_with_centroids_and_ellipses(
        normalized, y_complete, methods_ordered,
        FIGURES_DIR / 'figE27_picflu_pca_ellipses',
        'E27: PICFLU PCA with Centroids and 95% Confidence Ellipses',
        group_labels=('No PrAHRF/Death', 'PrAHRF/Death'))
    print("  Saved: figE27_picflu_pca_ellipses.pdf")

    # Best centroid summary
    best_centroid = picflu_centroid_df.loc[
        picflu_centroid_df['hotelling_permutation_p'].idxmin()]
    print(f"  Best Hotelling's T²: {best_centroid['norm_method']}/"
          f"{best_centroid['preprocessing']} "
          f"(p={best_centroid['hotelling_permutation_p']:.4f})")

    # --- Total Protein as dilution indicator ---
    print("\n--- Total Protein as Dilution Indicator ---")
    print(f"Total Protein: mean={np.mean(tp_complete):.0f}, "
          f"median={np.median(tp_complete):.0f}, "
          f"SD={np.std(tp_complete):.0f}, "
          f"CV={np.std(tp_complete)/np.mean(tp_complete)*100:.1f}%")

    # Correlation between total protein and analyte concentrations
    tp_corrs = []
    for j, analyte in enumerate(core_analytes):
        r, p = stats.pearsonr(tp_complete, X_complete[:, j])
        tp_corrs.append({'analyte': analyte, 'r_total_protein': r, 'p_value': p})
    tp_corr_df = pd.DataFrame(tp_corrs)
    tp_corr_df.to_csv(RESULTS_DIR / 'picflu_total_protein_correlations.csv', index=False)
    print(f"Analytes correlated with Total Protein (p<0.05): "
          f"{(tp_corr_df['p_value'] < 0.05).sum()}/{len(tp_corr_df)}")
    print(f"Mean |r| with Total Protein: {tp_corr_df['r_total_protein'].abs().mean():.3f}")

    # --- Summary statistics ---
    pairwise_jaccards = []
    for i, m1 in enumerate(methods_ordered):
        for j, m2 in enumerate(methods_ordered):
            if i < j:
                pairwise_jaccards.append(jaccard_matrix[i, j])

    summary = {
        'n_subjects': int(n_subjects),
        'n_analytes': len(core_analytes),
        'n_group0': int((y_complete == 0).sum()),
        'n_group1': int((y_complete == 1).sum()),
        'total_protein_cv': float(np.std(tp_complete) / np.mean(tp_complete) * 100),
        'mean_pairwise_jaccard': float(np.mean(pairwise_jaccards)) if pairwise_jaccards else 0,
        'min_pairwise_jaccard': float(np.min(pairwise_jaccards)) if pairwise_jaccards else 0,
        'max_pairwise_jaccard': float(np.max(pairwise_jaccards)) if pairwise_jaccards else 0,
    }

    # Add distance metric results to summary
    if len(picflu_distance_df) > 0:
        best = picflu_distance_df.loc[picflu_distance_df['permanova_r2'].idxmax()]
        summary['best_distance_norm'] = best['norm_method']
        summary['best_distance_metric'] = best['distance_metric']
        summary['best_distance_r2'] = float(best['permanova_r2'])
        summary['best_distance_p'] = float(best['permanova_p'])

    import json
    with open(RESULTS_DIR / 'picflu_reanalysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean pairwise Jaccard similarity: {summary['mean_pairwise_jaccard']:.3f}")
    print(f"Range: [{summary['min_pairwise_jaccard']:.3f}, {summary['max_pairwise_jaccard']:.3f}]")
    print(f"\nSaved: figE4_picflu_reanalysis.pdf, picflu_reanalysis_results.csv")
    print("=" * 60)

    return combined_df, summary


def load_juss_balf_data():
    """Load and clean absolute BALF data from Juss et al. 2016.

    Parses transposed sheets iii (ARDS) and iv (Controls) into a
    subjects×analytes matrix.

    Returns
    -------
    X : ndarray (28×44)
    y : ndarray (1=ARDS, 0=Ctrl)
    protein : ndarray (28,)
    analyte_names : list
    lloq_values : dict
    uloq_values : dict
    metadata : dict with 'gender' and 'severity'
    """
    xls_path = DATA_DIR / 'juss_supplementary_table_2.xlsx'

    # --- Sheet iii: BALF ARDS absolute ---
    df_ards = pd.read_excel(xls_path,
                            sheet_name='Table S2(iii) BALF, ARDS (abs.)',
                            header=None)
    # Row 3 = header (Identifier, LLoQ, ULoQ, A01..A18)
    # Row 4 = Gender, Row 5 = Severity, Row 6 = Protein
    # Rows 9-52 = 44 analytes; LLoQ col 1, ULoQ col 2, data cols 3:21
    ards_ids = df_ards.iloc[3, 3:21].tolist()  # A01..A18
    gender_ards = df_ards.iloc[4, 3:21].tolist()
    severity_ards = df_ards.iloc[5, 3:21].tolist()
    protein_ards = pd.to_numeric(df_ards.iloc[6, 3:21], errors='coerce').values

    analyte_names = df_ards.iloc[9:53, 0].tolist()
    lloq_raw = pd.to_numeric(df_ards.iloc[9:53, 1], errors='coerce').values
    uloq_raw = pd.to_numeric(df_ards.iloc[9:53, 2], errors='coerce').values

    # Data matrix: rows=analytes, cols=subjects → transpose to subjects×analytes
    X_ards_raw = df_ards.iloc[9:53, 3:21]
    X_ards_raw = X_ards_raw.apply(pd.to_numeric, errors='coerce')
    X_ards = X_ards_raw.values.T  # (18×44)

    # --- Sheet iv: BALF Ctrl absolute ---
    df_ctrl = pd.read_excel(xls_path,
                            sheet_name='Table S2(iv) BALF, Ctrl (abs.)',
                            header=None)
    # Row 3 = header (Identifier, LLoQ, ULoQ, C01..C10)
    # Row 4 = Protein
    # Rows 7-50 = 44 analytes
    ctrl_ids = df_ctrl.iloc[3, 3:13].tolist()  # C01..C10
    protein_ctrl = pd.to_numeric(df_ctrl.iloc[4, 3:13], errors='coerce').values

    X_ctrl_raw = df_ctrl.iloc[7:51, 3:13]
    X_ctrl_raw = X_ctrl_raw.apply(pd.to_numeric, errors='coerce')
    X_ctrl = X_ctrl_raw.values.T  # (10×44)

    # Combine
    X = np.vstack([X_ards, X_ctrl])  # (28×44)
    y = np.array([1]*18 + [0]*10)
    protein = np.concatenate([protein_ards, protein_ctrl])

    # Build LLoQ/ULoQ dicts
    lloq_values = {}
    uloq_values = {}
    for i, name in enumerate(analyte_names):
        if not np.isnan(lloq_raw[i]):
            lloq_values[name] = lloq_raw[i]
        if not np.isnan(uloq_raw[i]):
            uloq_values[name] = uloq_raw[i]

    metadata = {
        'gender': gender_ards,
        'severity': severity_ards,
        'ards_ids': ards_ids,
        'ctrl_ids': ctrl_ids,
    }

    print(f"Loaded Juss BALF data: {X.shape[0]} subjects ({(y==1).sum()} ARDS, "
          f"{(y==0).sum()} Ctrl), {X.shape[1]} analytes")
    print(f"  Analytes with LLoQ: {len(lloq_values)}/44")
    print(f"  Total protein CV: {np.nanstd(protein)/np.nanmean(protein)*100:.1f}%")

    return X, y, protein, analyte_names, lloq_values, uloq_values, metadata


def load_juss_protein_corrected():
    """Load author's protein-corrected BALF data from Juss et al. 2016.

    Parses sheets v (ARDS corr.) and vi (Ctrl corr.).

    Returns
    -------
    X_corr : ndarray (28×44)
    corr_analyte_names : list
    """
    xls_path = DATA_DIR / 'juss_supplementary_table_2.xlsx'

    # --- Sheet v: BALF ARDS corrected ---
    df_ards = pd.read_excel(xls_path,
                            sheet_name='Table S2(v) BALF, ARDS (corr.)',
                            header=None)
    # Row 3 = header (Identifier, A01..A18) — 19 cols, data cols 1:19
    # Rows 8-51 = 44 analytes
    corr_analyte_names = df_ards.iloc[8:52, 0].tolist()
    X_ards_corr = df_ards.iloc[8:52, 1:19]
    X_ards_corr = X_ards_corr.apply(pd.to_numeric, errors='coerce')
    X_ards_corr = X_ards_corr.values.T  # (18×44)

    # --- Sheet vi: BALF Ctrl corrected ---
    df_ctrl = pd.read_excel(xls_path,
                            sheet_name='Table S2(vi) BALF, Ctrl (corr.)',
                            header=None)
    # Row 3 = header (Identifier, C01..C10) — 11 cols, data cols 1:11
    # Rows 6-49 = 44 analytes
    X_ctrl_corr = df_ctrl.iloc[6:50, 1:11]
    X_ctrl_corr = X_ctrl_corr.apply(pd.to_numeric, errors='coerce')
    X_ctrl_corr = X_ctrl_corr.values.T  # (10×44)

    X_corr = np.vstack([X_ards_corr, X_ctrl_corr])  # (28×44)

    print(f"Loaded Juss protein-corrected data: {X_corr.shape}")

    return X_corr, corr_analyte_names


def load_juss_blood_data():
    """Load blood data from Juss et al. 2016 (negative control).

    Parses sheets i (Blood ARDS) and ii (Blood HVT).

    Returns
    -------
    X_blood : ndarray (36×44)
    y_blood : ndarray (1=ARDS, 0=HVT)
    blood_analytes : list
    lloq_blood : dict
    """
    xls_path = DATA_DIR / 'juss_supplementary_table_2.xlsx'

    # --- Sheet i: Blood ARDS ---
    df_ards = pd.read_excel(xls_path,
                            sheet_name='Table S2(i) Blood, ARDS',
                            header=None)
    # Row 3 = header, Row 4 = Gender, Row 5 = Severity
    # Rows 6-49 = 44 analytes; LLoQ col 1, ULoQ col 2, data cols 3:21
    blood_analytes = df_ards.iloc[6:50, 0].tolist()
    lloq_blood_raw = pd.to_numeric(df_ards.iloc[6:50, 1], errors='coerce').values

    X_ards_raw = df_ards.iloc[6:50, 3:21]
    # Handle "ND" strings → NaN
    X_ards_raw = X_ards_raw.apply(lambda col: col.map(
        lambda v: np.nan if isinstance(v, str) and v.strip().upper() == 'ND' else v
    ))
    X_ards_raw = X_ards_raw.apply(pd.to_numeric, errors='coerce')
    X_ards_blood = X_ards_raw.values.T  # (18×44)

    # --- Sheet ii: Blood HVT ---
    df_hvt = pd.read_excel(xls_path,
                           sheet_name='Table S2(ii) Blood, HVT',
                           header=None)
    # Row 3 = header; Rows 4-47 = 44 analytes; data cols 3:21
    X_hvt_raw = df_hvt.iloc[4:48, 3:21]
    X_hvt_raw = X_hvt_raw.apply(pd.to_numeric, errors='coerce')
    X_hvt_blood = X_hvt_raw.values.T  # (18×44)

    X_blood = np.vstack([X_ards_blood, X_hvt_blood])  # (36×44)
    y_blood = np.array([1]*18 + [0]*18)

    lloq_blood = {}
    for i, name in enumerate(blood_analytes):
        if not np.isnan(lloq_blood_raw[i]):
            lloq_blood[name] = lloq_blood_raw[i]

    print(f"Loaded Juss blood data: {X_blood.shape[0]} subjects "
          f"({(y_blood==1).sum()} ARDS, {(y_blood==0).sum()} HVT), "
          f"{X_blood.shape[1]} analytes")
    nd_count = np.isnan(X_ards_blood).sum()
    print(f"  ND values in blood ARDS: {nd_count}")

    return X_blood, y_blood, blood_analytes, lloq_blood


def _prepare_juss_matrix(X, analyte_names, lloq_values, max_censored_frac=0.80):
    """Apply LOD handling, filter heavily censored analytes, ensure positivity.

    Parameters
    ----------
    X : ndarray (n_subjects × n_analytes)
    analyte_names : list
    lloq_values : dict {analyte_name: lloq_value}
    max_censored_frac : float
        Exclude analytes with > this fraction at/below LLoQ/sqrt(2).

    Returns
    -------
    X_filtered : ndarray
    kept_names : list
    kept_idx : list of int
    """
    n_subjects, n_analytes = X.shape
    X_lod = X.copy()

    # LOD substitution: values below LLoQ → LLoQ/sqrt(2)
    for j, name in enumerate(analyte_names):
        if name in lloq_values:
            lloq = lloq_values[name]
            below = (X_lod[:, j] < lloq) | np.isnan(X_lod[:, j])
            X_lod[below, j] = lloq / np.sqrt(2)

    # Filter: exclude analytes with >max_censored_frac at/below LLoQ/sqrt(2)
    kept_idx = []
    for j, name in enumerate(analyte_names):
        if name in lloq_values:
            threshold = lloq_values[name] / np.sqrt(2)
            frac_censored = np.mean(X_lod[:, j] <= threshold + 1e-10)
            if frac_censored <= max_censored_frac:
                kept_idx.append(j)
        else:
            # No LLoQ (e.g. rlu analytes) — keep
            kept_idx.append(j)

    kept_names = [analyte_names[j] for j in kept_idx]
    X_filtered = X_lod[:, kept_idx]

    # Ensure positivity: replace zeros/negatives with min_positive/2
    min_pos = np.nanmin(X_filtered[X_filtered > 0])
    X_filtered[X_filtered <= 0] = min_pos / 2
    X_filtered[np.isnan(X_filtered)] = min_pos / 2

    # Impute any remaining NaN with column medians
    for j in range(X_filtered.shape[1]):
        col = X_filtered[:, j]
        nans = np.isnan(col)
        if nans.any():
            col[nans] = np.nanmedian(col)
            X_filtered[:, j] = col

    return X_filtered, kept_names, kept_idx


def run_juss_reanalysis():
    """Reanalysis pipeline for Juss et al. 2016 BALF data."""
    print("\n" + "=" * 60)
    print("JUSS ET AL. (AJRCCM 2016) REANALYSIS")
    print("BALF Biomarkers in ARDS vs Controls")
    print("=" * 60)

    # ---- Load data ----
    X, y, protein, analyte_names, lloq_values, uloq_values, metadata = \
        load_juss_balf_data()
    X_corr, corr_analyte_names = load_juss_protein_corrected()
    X_blood, y_blood, blood_analytes, lloq_blood = load_juss_blood_data()

    # ---- 4a/4b/4c: LOD handling + filtering for BALF ----
    X_filtered, kept_names, kept_idx = _prepare_juss_matrix(
        X, analyte_names, lloq_values, max_censored_frac=0.80)
    print(f"\nBALF: {len(kept_names)}/{len(analyte_names)} analytes retained "
          f"after LOD filtering")

    # Apply same filter to protein-corrected matrix
    X_corr_filtered = X_corr[:, kept_idx].copy()
    # Ensure positivity for corrected data
    min_pos = np.nanmin(X_corr_filtered[X_corr_filtered > 0])
    X_corr_filtered[X_corr_filtered <= 0] = min_pos / 2
    X_corr_filtered[np.isnan(X_corr_filtered)] = min_pos / 2

    # ---- 4d: Apply 7 normalizations ----
    print("\nApplying normalization methods (6 computational + protein correction)...")
    normalized = apply_normalizations(X_filtered, kept_names)
    normalized['protein_corrected'] = X_corr_filtered

    # Per-analyte normality testing
    print("\n--- Per-Analyte Normality Testing ---")
    normality_df_juss = test_analyte_normality(X_filtered, kept_names)
    normality_df_juss.to_csv(RESULTS_DIR / 'juss_normality_tests.csv', index=False)
    n_normal = (normality_df_juss['classification'] == 'normal').sum()
    n_nonnormal = (normality_df_juss['classification'] == 'non-normal').sum()
    print(f"  Normal: {n_normal}, Non-normal: {n_nonnormal}")

    # ---- 4e: Differential analysis (ARDS vs Control) ----
    print("\nPerforming differential analysis...")
    methods_ordered = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile',
                       'protein_corrected']
    all_results = {}
    sig_sets = {}

    for method_name in methods_ordered:
        X_norm = normalized[method_name]
        res_df = differential_analysis(X_norm, y)
        res_df['analyte'] = kept_names
        res_df['method'] = method_name

        # FDR correction
        valid_mask = ~res_df['p_value'].isna()
        if valid_mask.any():
            p_valid = res_df.loc[valid_mask, 'p_value'].values
            p_fdr, sig_fdr = multiple_testing_correction(p_valid, 'fdr_bh')
            res_df.loc[valid_mask, 'p_fdr'] = p_fdr
            res_df.loc[valid_mask, 'sig_fdr'] = sig_fdr
            res_df['sig_nominal'] = res_df['p_value'] < 0.05
        else:
            res_df['p_fdr'] = np.nan
            res_df['sig_fdr'] = False
            res_df['sig_nominal'] = False

        all_results[method_name] = res_df
        sig_sets[method_name] = set(
            res_df.loc[res_df['sig_nominal'] == True, 'analyte'].tolist()
        )

        n_sig_nom = res_df['sig_nominal'].sum()
        n_sig_fdr = res_df['sig_fdr'].sum() if 'sig_fdr' in res_df else 0
        print(f"  {method_name}: {n_sig_nom} nominal, {n_sig_fdr} FDR")

    # Combine results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_df.to_csv(RESULTS_DIR / 'juss_reanalysis_results.csv', index=False)

    # ---- 4f: Figures ----

    # === Figure E12: Overview (mirrors E4) ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Significant analytes bar chart
    ax = axes[0]
    n_sig_nominal = [combined_df[(combined_df['method'] == m) &
                     (combined_df['sig_nominal'] == True)].shape[0]
                     for m in methods_ordered]
    n_sig_fdr = [combined_df[(combined_df['method'] == m) &
                 (combined_df['sig_fdr'] == True)].shape[0]
                 for m in methods_ordered]

    x_pos = np.arange(len(methods_ordered))
    width = 0.35
    ax.bar(x_pos - width/2, n_sig_nominal, width, label='Nominal (p < 0.05)',
           color='#D55E00', alpha=0.8)
    ax.bar(x_pos + width/2, n_sig_fdr, width, label='FDR-corrected',
           color='#0072B2', alpha=0.8)
    ax.set_xlabel('Normalization Method')
    ax.set_ylabel('Number of Significant Analytes')
    ax.set_title('A) Significant Analytes by Method', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods_ordered, rotation=45, ha='right')
    ax.legend(fontsize=9)

    # Panel B: Significance concordance heatmap
    ax = axes[1]
    sig_matrix = pd.DataFrame(index=kept_names, columns=methods_ordered, data=False)
    for method in methods_ordered:
        res = all_results[method]
        for _, row in res.iterrows():
            if row['sig_nominal']:
                sig_matrix.loc[row['analyte'], method] = True
    sig_numeric = sig_matrix.astype(int)
    any_sig = sig_numeric.sum(axis=1) > 0
    if any_sig.any():
        sig_show = sig_numeric.loc[any_sig]
        sns.heatmap(sig_show, cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'Significant'},
                    linewidths=0.5, linecolor='gray')
        ax.set_title('B) Significance Concordance (p < 0.05)', fontweight='bold')
        ax.set_xlabel('Normalization Method')
        ax.set_ylabel('Analyte')
    else:
        ax.text(0.5, 0.5, 'No significant\nanalytes found',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('B) Significance Concordance', fontweight='bold')

    # Panel C: Pairwise Jaccard similarity
    ax = axes[2]
    jaccard_matrix = np.zeros((len(methods_ordered), len(methods_ordered)))
    for i, m1 in enumerate(methods_ordered):
        for j, m2 in enumerate(methods_ordered):
            s1 = sig_sets.get(m1, set())
            s2 = sig_sets.get(m2, set())
            union = s1 | s2
            if len(union) > 0:
                jaccard_matrix[i, j] = len(s1 & s2) / len(union)
            else:
                jaccard_matrix[i, j] = 1.0 if i == j else 0.0

    jaccard_df = pd.DataFrame(jaccard_matrix, index=methods_ordered,
                              columns=methods_ordered)
    sns.heatmap(jaccard_df, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                vmin=0, vmax=1, linewidths=0.5, square=True)
    ax.set_title('C) Pairwise Jaccard Similarity', fontweight='bold')

    fig.suptitle('Figure E12: Juss et al. BALF Data Reanalysis (ARDS vs Control)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE12_juss_reanalysis.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE12_juss_reanalysis.png')
    plt.close(fig)
    print("  Saved: figE12_juss_reanalysis.pdf")

    # === Figure E13: Volcano plots (2×4) ===
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for idx, method in enumerate(methods_ordered):
        ax = axes[idx // 4, idx % 4]
        res = all_results[method].copy()
        res = res.dropna(subset=['p_value', 'fold_change'])
        res['log2fc'] = np.log2(res['fold_change'].clip(lower=1e-10))
        res['neg_log10p'] = -np.log10(res['p_value'].clip(lower=1e-300))

        sig_mask = res['sig_nominal'] == True
        ax.scatter(res.loc[~sig_mask, 'log2fc'], res.loc[~sig_mask, 'neg_log10p'],
                   c='#999999', alpha=0.6, s=20, edgecolors='none', label='NS')
        ax.scatter(res.loc[sig_mask, 'log2fc'], res.loc[sig_mask, 'neg_log10p'],
                   c='#D55E00', alpha=0.8, s=30, edgecolors='none', label='p < 0.05')

        top5 = res.nsmallest(5, 'p_value')
        for _, row in top5.iterrows():
            label = str(row['analyte']).replace('BALF ', '')
            if len(label) > 15:
                label = label[:12] + '...'
            ax.annotate(label, (row['log2fc'], row['neg_log10p']),
                        fontsize=6, ha='center', va='bottom')

        ax.axhline(-np.log10(0.05), color='black', linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.set_xlabel('log₂(Fold Change)')
        ax.set_ylabel('-log₁₀(p-value)')
        ax.set_title(f'{method}', fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=7)

    # 8th panel: scatter CLR vs protein_corrected -log10(p)
    ax = axes[1, 3]
    res_clr = all_results['clr'].set_index('analyte')
    res_pc = all_results['protein_corrected'].set_index('analyte')
    common = res_clr.index.intersection(res_pc.index)
    x_vals = -np.log10(res_clr.loc[common, 'p_value'].clip(lower=1e-300))
    y_vals = -np.log10(res_pc.loc[common, 'p_value'].clip(lower=1e-300))
    ax.scatter(x_vals, y_vals, c='#009E73', alpha=0.7, s=30, edgecolors='none')
    max_val = max(x_vals.max(), y_vals.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=0.8)
    ax.axhline(-np.log10(0.05), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(-np.log10(0.05), color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('-log₁₀(p) CLR')
    ax.set_ylabel('-log₁₀(p) Protein Corrected')
    ax.set_title('CLR vs Protein Corr.', fontweight='bold')

    fig.suptitle('Figure E13: Volcano Plots (Juss et al., ARDS vs Control)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE13_juss_volcanos.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE13_juss_volcanos.png')
    plt.close(fig)
    print("  Saved: figE13_juss_volcanos.pdf")

    # === Figure E14: PCA biplots (log-transformed + scaled, 2×4) ===
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for idx, method in enumerate(methods_ordered):
        ax = axes[idx // 4, idx % 4]
        X_prep = prepare_pca_data(normalized[method], method,
                                  log_transform=True, scale=True)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_prep)

        for g, label, color in [(0, 'Control', '#0072B2'), (1, 'ARDS', '#D55E00')]:
            mask = y == g
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.7,
                       s=40, edgecolors='none', label=label)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        preproc = 'scaled' if method == 'clr' else 'log + scaled'
        ax.set_title(f'{method} ({preproc})', fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=8)

    # 8th panel: protein-colored PCA on CLR data (scaled)
    ax = axes[1, 3]
    X_clr_scaled = prepare_pca_data(normalized['clr'], 'clr',
                                    log_transform=False, scale=True)
    pca_clr = PCA(n_components=2)
    X_pca_clr = pca_clr.fit_transform(X_clr_scaled)
    sc = ax.scatter(X_pca_clr[:, 0], X_pca_clr[:, 1], c=protein,
                    cmap='viridis', alpha=0.7, s=40, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Total Protein (µg/mL)')
    ax.set_xlabel(f'PC1 ({pca_clr.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_clr.explained_variance_ratio_[1]:.1%})')
    ax.set_title('CLR scaled (protein colored)', fontweight='bold')

    fig.suptitle('Figure E14: PCA Biplots (Juss et al., ARDS vs Control)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE14_juss_pca.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE14_juss_pca.png')
    plt.close(fig)
    print("  Saved: figE14_juss_pca.pdf")

    # === Figure E14b: PCA Preprocessing Comparison (Juss) ===
    preproc_configs = [
        ('Raw', False, False),
        ('Log-transformed', True, False),
        ('Centered + Scaled', False, True),
        ('Log + Centered + Scaled', True, True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    # Row 0: "none" normalization; Row 1: "clr" normalization
    for row, (norm_method, norm_label) in enumerate([('none', 'No Normalization'),
                                                      ('clr', 'CLR')]):
        for col, (preproc_label, do_log, do_scale) in enumerate(preproc_configs):
            ax = axes[row, col]
            X_prep = prepare_pca_data(normalized[norm_method], norm_method,
                                      log_transform=do_log, scale=do_scale)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_prep)

            for g, label, color in [(0, 'Control', '#0072B2'),
                                     (1, 'ARDS', '#D55E00')]:
                mask = y == g
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.7,
                           s=40, edgecolors='none', label=label)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title(f'{norm_label}: {preproc_label}', fontweight='bold',
                         fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    fig.suptitle('Figure E14b: PCA Preprocessing Comparison (Juss)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE14b_juss_pca_preprocessing.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE14b_juss_pca_preprocessing.png')
    plt.close(fig)
    print("  Saved: figE14b_juss_pca_preprocessing.pdf")

    # === Figure E15: Rank comparison (1×2) ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    rank_matrix = pd.DataFrame(index=kept_names)
    for method in methods_ordered:
        res = all_results[method].copy().set_index('analyte')
        rank_matrix[method] = res['test_stat'].abs().rank(ascending=False)

    # Panel A: Spearman rank correlation
    ax = axes[0]
    rank_corr = rank_matrix.corr(method='spearman')
    sns.heatmap(rank_corr, ax=ax, annot=True, fmt='.2f', cmap='Blues',
                vmin=0, vmax=1, linewidths=0.5, square=True)
    ax.set_title('A) Rank Correlation Across Methods', fontweight='bold')

    # Panel B: Top-10 analyte ranks across methods
    ax = axes[1]
    raw_top10 = rank_matrix['none'].nsmallest(10).index.tolist()
    rank_top10 = rank_matrix.loc[raw_top10, methods_ordered]
    short_names = [n.replace('BALF ', '')[:18] for n in rank_top10.index]
    rank_display = rank_top10.copy()
    rank_display.index = short_names
    sns.heatmap(rank_display, ax=ax, annot=True, fmt='.0f', cmap='YlOrRd_r',
                linewidths=0.5, cbar_kws={'label': 'Rank'})
    ax.set_title('B) Top-10 Analyte Ranks Across Methods', fontweight='bold')
    ax.set_ylabel('Analyte (ranked by raw analysis)')
    ax.set_xlabel('Normalization Method')

    fig.suptitle('Figure E15: Analyte Ranking Comparison (Juss et al.)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE15_juss_ranks.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE15_juss_ranks.png')
    plt.close(fig)
    print("  Saved: figE15_juss_ranks.pdf")

    # --- BALF Distance metric analysis ---
    print("\n--- BALF Distance Metric Analysis (PERMANOVA R²) ---")
    juss_balf_distance_df = distance_metric_analysis(
        normalized, y, methods_ordered, n_permutations=999)
    juss_balf_distance_df.to_csv(
        RESULTS_DIR / 'juss_distance_results_balf.csv', index=False)
    print(f"  Computed {len(juss_balf_distance_df)} norm×metric combinations")

    if len(juss_balf_distance_df) > 0:
        best = juss_balf_distance_df.loc[
            juss_balf_distance_df['permanova_r2'].idxmax()]
        print(f"  Best BALF: {best['norm_method']}/{best['distance_metric']} "
              f"R²={best['permanova_r2']:.3f} (p={best['permanova_p']:.3f})")

    # --- BALF Log transform comparison ---
    print("\n--- BALF Log Transform Comparison ---")
    log_variants_balf = add_log_variants(normalized)
    juss_log_df = plot_log_comparison(
        normalized, log_variants_balf, y, kept_names,
        FIGURES_DIR / 'figE21_juss_log_comparison',
        'E21: Juss BALF Log Transform Comparison', n_permutations=999)
    juss_log_df.to_csv(RESULTS_DIR / 'juss_log_comparison_balf.csv', index=False)
    print("  Saved: figE21_juss_log_comparison.pdf, juss_log_comparison_balf.csv")

    # --- Stratified log comparison (by analyte distribution) ---
    print("\n--- Stratified Log Comparison (by Analyte Distribution) ---")
    juss_log_strat_df = plot_log_comparison_stratified(
        normalized, log_variants_balf, y, kept_names, normality_df_juss,
        FIGURES_DIR / 'figE21b_juss_log_stratified',
        'E21b: Juss Log Comparison (by Analyte Distribution)')
    print("  Saved: figE21b_juss_log_stratified.pdf")

    # --- BALF PCA Centroid Comparison ---
    print("\n--- BALF PCA Centroid Comparison ---")
    juss_balf_centroid_df = pca_centroid_comparison_realdata(
        normalized, y, methods_ordered,
        n_components=2, n_permutations=999)
    juss_balf_centroid_df.to_csv(
        RESULTS_DIR / 'juss_pca_centroid_results_balf.csv', index=False)
    print(f"  Computed {len(juss_balf_centroid_df)} norm×preprocessing combinations")

    plot_pca_centroid_heatmap(
        juss_balf_centroid_df,
        FIGURES_DIR / 'figE28_juss_pca_centroid',
        'E28: Juss BALF PCA Centroid Comparison')
    print("  Saved: figE28_juss_pca_centroid.pdf")

    plot_pca_with_centroids_and_ellipses(
        normalized, y, methods_ordered,
        FIGURES_DIR / 'figE29_juss_pca_ellipses',
        'E29: Juss BALF PCA with Centroids and 95% Confidence Ellipses',
        group_labels=('Control', 'ARDS'))
    print("  Saved: figE29_juss_pca_ellipses.pdf")

    best_balf_centroid = juss_balf_centroid_df.loc[
        juss_balf_centroid_df['hotelling_permutation_p'].idxmin()]
    print(f"  Best BALF Hotelling's T²: {best_balf_centroid['norm_method']}/"
          f"{best_balf_centroid['preprocessing']} "
          f"(p={best_balf_centroid['hotelling_permutation_p']:.4f})")

    # === Figure E16: Protein correction comparison (1×3) ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    res_pc = all_results['protein_corrected'].set_index('analyte')
    res_clr = all_results['clr'].set_index('analyte')
    res_pqn = all_results['pqn'].set_index('analyte')
    common_all = res_pc.index.intersection(res_clr.index).intersection(res_pqn.index)

    # Panel A: Scatter protein_corrected vs CLR
    ax = axes[0]
    x_v = -np.log10(res_pc.loc[common_all, 'p_value'].clip(lower=1e-300))
    y_v = -np.log10(res_clr.loc[common_all, 'p_value'].clip(lower=1e-300))
    ax.scatter(x_v, y_v, c='#009E73', alpha=0.7, s=30, edgecolors='none')
    max_v = max(x_v.max(), y_v.max()) * 1.1
    ax.plot([0, max_v], [0, max_v], 'k--', alpha=0.4)
    ax.axhline(-np.log10(0.05), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(-np.log10(0.05), color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('-log₁₀(p) Protein Corrected')
    ax.set_ylabel('-log₁₀(p) CLR')
    ax.set_title('A) Protein Corr. vs CLR', fontweight='bold')

    # Panel B: Scatter protein_corrected vs PQN
    ax = axes[1]
    y_v2 = -np.log10(res_pqn.loc[common_all, 'p_value'].clip(lower=1e-300))
    ax.scatter(x_v, y_v2, c='#CC79A7', alpha=0.7, s=30, edgecolors='none')
    max_v2 = max(x_v.max(), y_v2.max()) * 1.1
    ax.plot([0, max_v2], [0, max_v2], 'k--', alpha=0.4)
    ax.axhline(-np.log10(0.05), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(-np.log10(0.05), color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('-log₁₀(p) Protein Corrected')
    ax.set_ylabel('-log₁₀(p) PQN')
    ax.set_title('B) Protein Corr. vs PQN', fontweight='bold')

    # Panel C: Significance overlap bar chart
    ax = axes[2]
    sig_pc = sig_sets.get('protein_corrected', set())
    sig_clr = sig_sets.get('clr', set())
    shared = sig_pc & sig_clr
    unique_pc = sig_pc - sig_clr
    unique_clr = sig_clr - sig_pc

    bars = ['Protein Corr.\nonly', 'Shared', 'CLR only']
    vals = [len(unique_pc), len(shared), len(unique_clr)]
    colors = ['#E69F00', '#56B4E9', '#009E73']
    ax.bar(bars, vals, color=colors, alpha=0.8)
    ax.set_ylabel('Number of Significant Analytes')
    ax.set_title('C) Significance Overlap: Protein Corr. vs CLR', fontweight='bold')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, str(v), ha='center', fontweight='bold')

    fig.suptitle('Figure E16: Protein Correction Comparison (Juss et al.)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE16_juss_protein_comparison.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE16_juss_protein_comparison.png')
    plt.close(fig)
    print("  Saved: figE16_juss_protein_comparison.pdf")

    # ---- 4g: Total Protein Analysis ----
    print("\n--- Total Protein as Dilution Indicator ---")
    protein_ards = protein[y == 1]
    protein_ctrl = protein[y == 0]
    print(f"Total Protein (combined): mean={np.nanmean(protein):.0f}, "
          f"SD={np.nanstd(protein):.0f}, "
          f"CV={np.nanstd(protein)/np.nanmean(protein)*100:.1f}%")
    print(f"  ARDS: mean={np.nanmean(protein_ards):.0f}, "
          f"CV={np.nanstd(protein_ards)/np.nanmean(protein_ards)*100:.1f}%")
    print(f"  Ctrl: mean={np.nanmean(protein_ctrl):.0f}, "
          f"CV={np.nanstd(protein_ctrl)/np.nanmean(protein_ctrl)*100:.1f}%")

    tp_corrs = []
    for j, analyte in enumerate(kept_names):
        r, p_val = stats.pearsonr(protein, X_filtered[:, j])
        tp_corrs.append({'analyte': analyte, 'r_total_protein': r, 'p_value': p_val})
    tp_corr_df = pd.DataFrame(tp_corrs)
    tp_corr_df.to_csv(RESULTS_DIR / 'juss_total_protein_correlations.csv', index=False)
    print(f"Analytes correlated with Total Protein (p<0.05): "
          f"{(tp_corr_df['p_value'] < 0.05).sum()}/{len(tp_corr_df)}")
    print(f"Mean |r| with Total Protein: {tp_corr_df['r_total_protein'].abs().mean():.3f}")

    # ---- 4h: Blood Negative Control ----
    print("\n--- Blood Negative Control ---")
    X_blood_filtered, blood_kept_names, blood_kept_idx = _prepare_juss_matrix(
        X_blood, blood_analytes, lloq_blood, max_censored_frac=0.80)
    print(f"Blood: {len(blood_kept_names)}/{len(blood_analytes)} analytes retained")

    blood_normalized = apply_normalizations(X_blood_filtered, blood_kept_names)
    blood_methods = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile']
    blood_results = {}
    blood_sig_sets = {}

    for method_name in blood_methods:
        X_norm = blood_normalized[method_name]
        res_df = differential_analysis(X_norm, y_blood)
        res_df['analyte'] = blood_kept_names
        res_df['method'] = method_name

        valid_mask = ~res_df['p_value'].isna()
        if valid_mask.any():
            p_valid = res_df.loc[valid_mask, 'p_value'].values
            p_fdr, sig_fdr = multiple_testing_correction(p_valid, 'fdr_bh')
            res_df.loc[valid_mask, 'p_fdr'] = p_fdr
            res_df.loc[valid_mask, 'sig_fdr'] = sig_fdr
            res_df['sig_nominal'] = res_df['p_value'] < 0.05
        else:
            res_df['p_fdr'] = np.nan
            res_df['sig_fdr'] = False
            res_df['sig_nominal'] = False

        blood_results[method_name] = res_df
        blood_sig_sets[method_name] = set(
            res_df.loc[res_df['sig_nominal'] == True, 'analyte'].tolist()
        )

        n_sig = res_df['sig_nominal'].sum()
        print(f"  Blood {method_name}: {n_sig} nominal significant")

    # Blood distance metric analysis
    print("\n--- Blood Distance Metric Analysis (PERMANOVA R²) ---")
    juss_blood_distance_df = distance_metric_analysis(
        blood_normalized, y_blood, blood_methods, n_permutations=999)
    juss_blood_distance_df.to_csv(
        RESULTS_DIR / 'juss_distance_results_blood.csv', index=False)
    print(f"  Computed {len(juss_blood_distance_df)} norm×metric combinations")

    if len(juss_blood_distance_df) > 0:
        best = juss_blood_distance_df.loc[
            juss_blood_distance_df['permanova_r2'].idxmax()]
        print(f"  Best Blood: {best['norm_method']}/{best['distance_metric']} "
              f"R²={best['permanova_r2']:.3f} (p={best['permanova_p']:.3f})")

    # Blood PCA Centroid Comparison
    print("\n--- Blood PCA Centroid Comparison ---")
    juss_blood_centroid_df = pca_centroid_comparison_realdata(
        blood_normalized, y_blood, blood_methods,
        n_components=2, n_permutations=999)
    juss_blood_centroid_df.to_csv(
        RESULTS_DIR / 'juss_pca_centroid_results_blood.csv', index=False)
    print(f"  Computed {len(juss_blood_centroid_df)} norm×preprocessing combinations")

    best_blood_centroid = juss_blood_centroid_df.loc[
        juss_blood_centroid_df['hotelling_permutation_p'].idxmin()]
    print(f"  Best Blood Hotelling's T²: {best_blood_centroid['norm_method']}/"
          f"{best_blood_centroid['preprocessing']} "
          f"(p={best_blood_centroid['hotelling_permutation_p']:.4f})")

    # BALF vs Blood comparison heatmap
    plot_pca_centroid_heatmap_comparison(
        juss_balf_centroid_df, juss_blood_centroid_df,
        FIGURES_DIR / 'figE28b_juss_pca_centroid_balf_vs_blood',
        'E28b: Juss PCA Centroid — BALF vs Blood')
    print("  Saved: figE28b_juss_pca_centroid_balf_vs_blood.pdf")

    # Blood Jaccard matrix
    blood_jaccard = np.zeros((len(blood_methods), len(blood_methods)))
    for i, m1 in enumerate(blood_methods):
        for j, m2 in enumerate(blood_methods):
            s1 = blood_sig_sets.get(m1, set())
            s2 = blood_sig_sets.get(m2, set())
            union = s1 | s2
            if len(union) > 0:
                blood_jaccard[i, j] = len(s1 & s2) / len(union)
            else:
                blood_jaccard[i, j] = 1.0 if i == j else 0.0

    # BALF Jaccard (6 computational methods only, for comparison)
    balf_methods_6 = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile']
    balf_jaccard_6 = np.zeros((len(balf_methods_6), len(balf_methods_6)))
    for i, m1 in enumerate(balf_methods_6):
        for j, m2 in enumerate(balf_methods_6):
            s1 = sig_sets.get(m1, set())
            s2 = sig_sets.get(m2, set())
            union = s1 | s2
            if len(union) > 0:
                balf_jaccard_6[i, j] = len(s1 & s2) / len(union)
            else:
                balf_jaccard_6[i, j] = 1.0 if i == j else 0.0

    mean_balf_j = balf_jaccard_6[np.triu_indices(len(balf_methods_6), k=1)].mean()
    mean_blood_j = blood_jaccard[np.triu_indices(len(blood_methods), k=1)].mean()
    print(f"\nMean Jaccard (BALF, 6 methods): {mean_balf_j:.3f}")
    print(f"Mean Jaccard (Blood, 6 methods): {mean_blood_j:.3f}")
    print(f"Blood concordance {'>' if mean_blood_j > mean_balf_j else '<'} "
          f"BALF concordance (prediction: blood > BALF)")

    # === Figure E17: Blood negative control (1×2) ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Significant analytes by method (blood)
    ax = axes[0]
    blood_combined = pd.concat(blood_results.values(), ignore_index=True)
    n_sig_blood = [blood_combined[(blood_combined['method'] == m) &
                   (blood_combined['sig_nominal'] == True)].shape[0]
                   for m in blood_methods]
    n_sig_blood_fdr = [blood_combined[(blood_combined['method'] == m) &
                       (blood_combined['sig_fdr'] == True)].shape[0]
                       for m in blood_methods]

    x_pos = np.arange(len(blood_methods))
    ax.bar(x_pos - width/2, n_sig_blood, width, label='Nominal (p < 0.05)',
           color='#D55E00', alpha=0.8)
    ax.bar(x_pos + width/2, n_sig_blood_fdr, width, label='FDR-corrected',
           color='#0072B2', alpha=0.8)
    ax.set_xlabel('Normalization Method')
    ax.set_ylabel('Number of Significant Analytes')
    ax.set_title('A) Blood: Significant Analytes by Method', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(blood_methods, rotation=45, ha='right')
    ax.legend(fontsize=9)

    # Panel B: Side-by-side Jaccard heatmaps
    ax = axes[1]
    # Use a combined display: BALF top-left, Blood bottom-right
    combined_size = len(balf_methods_6) + len(blood_methods)
    combined_jaccard = np.full((combined_size, combined_size), np.nan)
    combined_jaccard[:len(balf_methods_6), :len(balf_methods_6)] = balf_jaccard_6
    combined_jaccard[len(balf_methods_6):, len(balf_methods_6):] = blood_jaccard
    combined_labels = [f'BALF\n{m}' for m in balf_methods_6] + \
                      [f'Blood\n{m}' for m in blood_methods]

    mask = np.isnan(combined_jaccard)
    sns.heatmap(combined_jaccard, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                vmin=0, vmax=1, linewidths=0.5, square=True,
                xticklabels=combined_labels, yticklabels=combined_labels,
                mask=mask)
    ax.set_title(f'B) Jaccard: BALF (mean={mean_balf_j:.2f}) vs '
                 f'Blood (mean={mean_blood_j:.2f})', fontweight='bold')

    fig.suptitle('Figure E17: Blood Negative Control (Juss et al.)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE17_juss_blood_control.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE17_juss_blood_control.png')
    plt.close(fig)
    print("  Saved: figE17_juss_blood_control.pdf")

    # === Figure E19: Distance metric comparison (BALF + Blood) ===
    plot_permanova_heatmap(
        juss_balf_distance_df,
        FIGURES_DIR / 'figE19_juss_distance',
        'E19: Juss Distance Metric Analysis',
        blood_results=juss_blood_distance_df)
    print("  Saved: figE19_juss_distance.pdf")

    # ---- 4i: Save outputs ----
    import json

    pairwise_jaccards = jaccard_matrix[np.triu_indices(len(methods_ordered), k=1)]

    summary = {
        'n_subjects': int(X.shape[0]),
        'n_ards': int((y == 1).sum()),
        'n_ctrl': int((y == 0).sum()),
        'n_analytes_total': len(analyte_names),
        'n_analytes_retained': len(kept_names),
        'n_blood_subjects': int(X_blood.shape[0]),
        'n_blood_analytes_retained': len(blood_kept_names),
        'total_protein_cv_combined': float(
            np.nanstd(protein) / np.nanmean(protein) * 100),
        'total_protein_cv_ards': float(
            np.nanstd(protein_ards) / np.nanmean(protein_ards) * 100),
        'total_protein_cv_ctrl': float(
            np.nanstd(protein_ctrl) / np.nanmean(protein_ctrl) * 100),
        'mean_pairwise_jaccard_balf': float(np.mean(pairwise_jaccards)),
        'min_pairwise_jaccard_balf': float(np.min(pairwise_jaccards)),
        'max_pairwise_jaccard_balf': float(np.max(pairwise_jaccards)),
        'mean_pairwise_jaccard_blood': float(mean_blood_j),
        'mean_abs_r_protein': float(tp_corr_df['r_total_protein'].abs().mean()),
        'analytes_corr_protein_p05': int((tp_corr_df['p_value'] < 0.05).sum()),
    }

    # Add distance metric results to summary
    if len(juss_balf_distance_df) > 0:
        best_balf = juss_balf_distance_df.loc[
            juss_balf_distance_df['permanova_r2'].idxmax()]
        summary['best_balf_distance_norm'] = best_balf['norm_method']
        summary['best_balf_distance_metric'] = best_balf['distance_metric']
        summary['best_balf_distance_r2'] = float(best_balf['permanova_r2'])
    if len(juss_blood_distance_df) > 0:
        best_blood = juss_blood_distance_df.loc[
            juss_blood_distance_df['permanova_r2'].idxmax()]
        summary['best_blood_distance_norm'] = best_blood['norm_method']
        summary['best_blood_distance_metric'] = best_blood['distance_metric']
        summary['best_blood_distance_r2'] = float(best_blood['permanova_r2'])

    with open(RESULTS_DIR / 'juss_reanalysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean pairwise Jaccard (7 methods): {summary['mean_pairwise_jaccard_balf']:.3f}")
    print(f"Range: [{summary['min_pairwise_jaccard_balf']:.3f}, "
          f"{summary['max_pairwise_jaccard_balf']:.3f}]")
    print(f"\nSaved: juss_reanalysis_results.csv, juss_reanalysis_summary.json, "
          f"juss_total_protein_correlations.csv")
    print("=" * 60)

    return combined_df, summary


if __name__ == '__main__':
    run_picflu_reanalysis()
    run_juss_reanalysis()
