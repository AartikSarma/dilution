#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Biomarker Dilution Simulation Study.

Runs all analyses needed for the manuscript:
1. Dilution model comparison
2. Normalization method evaluation across dilution severities
3. Impact on univariate, multivariate, and ML analyses
4. Power analysis under dilution
5. Feature selection robustness
6. Sample size and effect size interaction

Outputs: figures/ directory with all plots, results/ directory with data tables.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import json
import warnings
import time
import shutil
import multiprocessing as mp

warnings.filterwarnings('ignore')

from biomarker_dilution_sim import (
    generate_dataset, generate_dilution_factors,
    generate_dilution_factors_time_dependent,
    generate_dilution_factors_covariate_dependent,
    normalize_data, simulate_batch_effects, normalize_combat,
    analyze_univariate, analyze_univariate_enhanced,
    analyze_correlation, analyze_pca, analyze_clustering,
    analyze_classification, analyze_classification_advanced,
    feature_selection, cross_validate_classification,
    multiple_testing_correction, calculate_effect_size,
    bootstrap_confidence_interval, power_analysis, sample_size_estimation,
    evaluate_univariate, evaluate_correlation, evaluate_pca,
    evaluate_clustering, evaluate_classification,
    run_single_simulation, apply_dilution, centered_log_ratio,
    compute_distance_matrix, evaluate_distance_matrix, permanova_r2,
)
from visualization_module import (
    plot_dilution_effect, plot_normalization_comparison,
    plot_volcano, plot_pca_3d, plot_forest,
    plot_heatmap_clustered, plot_power_curve,
    plot_dilution_distribution_comparison
)

# Setup output directories
FIGURES_DIR = Path('figures')
RESULTS_DIR = Path('results')
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Plotting defaults — publication quality
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Colorblind-safe palette
CB_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']


def analysis_1_dilution_models():
    """Compare different dilution factor distributions (Figure 1)."""
    print("=" * 60)
    print("Analysis 1: Dilution Model Comparison")
    print("=" * 60)

    np.random.seed(42)
    n_subjects = 500

    dilution_models = {
        'Beta(8,2)\n(Mild)': generate_dilution_factors(n_subjects, 8.0, 2.0, 'beta'),
        'Beta(5,5)\n(Moderate)': generate_dilution_factors(n_subjects, 5.0, 5.0, 'beta'),
        'Beta(2,8)\n(Severe)': generate_dilution_factors(n_subjects, 2.0, 8.0, 'beta'),
        'Bimodal': generate_dilution_factors(n_subjects, 5.0, 5.0, 'bimodal'),
        'Mixture': generate_dilution_factors(n_subjects, 5.0, 5.0, 'mixture'),
        'Uniform': generate_dilution_factors(n_subjects, 5.0, 5.0, 'uniform'),
    }

    # Figure 1: Dilution factor distributions
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#607D8B']

    for idx, (name, factors) in enumerate(dilution_models.items()):
        ax = axes[idx // 3, idx % 3]
        ax.hist(factors, bins=40, color=colors[idx], alpha=0.7, edgecolor='white', density=True)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Dilution Factor')
        ax.set_ylabel('Density')
        ax.axvline(np.mean(factors), color='black', linestyle='--', linewidth=1.5,
                    label=f'Mean={np.mean(factors):.2f}')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle('Figure 1: Dilution Factor Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_dilution_distributions.png')
    fig.savefig(FIGURES_DIR / 'fig1_dilution_distributions.pdf', format='pdf')
    plt.close(fig)

    # Statistics table
    stats_rows = []
    for name, factors in dilution_models.items():
        stats_rows.append({
            'Model': name.replace('\n', ' '),
            'Mean': f'{np.mean(factors):.3f}',
            'Std': f'{np.std(factors):.3f}',
            'Median': f'{np.median(factors):.3f}',
            'Min': f'{np.min(factors):.3f}',
            'Max': f'{np.max(factors):.3f}',
            'Skewness': f'{float(pd.Series(factors).skew()):.3f}',
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(RESULTS_DIR / 'table1_dilution_stats.csv', index=False)
    print(f"  Saved: fig1_dilution_distributions.png, table1_dilution_stats.csv")
    print(stats_df.to_string(index=False))
    return dilution_models


def analysis_2_dilution_impact_on_data():
    """Visualize how dilution distorts biomarker data (Figure 2)."""
    print("\n" + "=" * 60)
    print("Analysis 2: Dilution Impact on Biomarker Data")
    print("=" * 60)

    np.random.seed(42)

    params = {
        'n_subjects': 200,
        'n_biomarkers': 10,
        'n_groups': 2,
        'correlation_type': 'moderate',
        'effect_size': 0.8,
        'distribution': 'lognormal',
        'dilution_alpha': 2.0,
        'dilution_beta': 5.0,
        'lod_percentile': 0.1,
        'lod_handling': 'substitute',
    }

    dataset = generate_dataset(**params)
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    y = dataset['y']
    dilution_factors = dataset['dilution_factors']

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: True vs Observed scatter
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(X_true[:, 0], X_obs[:, 0], c=dilution_factors, cmap='viridis',
                     alpha=0.6, s=20, edgecolor='none')
    max_val = max(X_true[:, 0].max(), X_obs[:, 0].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=1.5, label='y=x')
    ax1.set_xlabel('True Concentration')
    ax1.set_ylabel('Observed Concentration')
    ax1.set_title('A) True vs. Observed (Biomarker 1)')
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('Dilution Factor')
    ax1.legend(loc='upper left')

    # Panel B: Group separation - True data
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca_true = pca.fit_transform(X_true)
    for g in range(2):
        mask = y == g
        ax2.scatter(X_pca_true[mask, 0], X_pca_true[mask, 1], alpha=0.5, s=20,
                    label=f'Group {g}')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax2.set_title('B) PCA - True Data')
    ax2.legend()

    # Panel C: Group separation - Observed data
    ax3 = fig.add_subplot(gs[0, 2])
    pca_obs = PCA(n_components=2)
    X_pca_obs = pca_obs.fit_transform(X_obs)
    for g in range(2):
        mask = y == g
        ax3.scatter(X_pca_obs[mask, 0], X_pca_obs[mask, 1], alpha=0.5, s=20,
                    label=f'Group {g}')
    ax3.set_xlabel(f'PC1 ({pca_obs.explained_variance_ratio_[0]:.1%} var)')
    ax3.set_ylabel(f'PC2 ({pca_obs.explained_variance_ratio_[1]:.1%} var)')
    ax3.set_title('C) PCA - Observed (Diluted) Data')
    ax3.legend()

    # Panel D: Correlation matrix - True
    ax4 = fig.add_subplot(gs[1, 0])
    corr_true = np.corrcoef(X_true.T)
    im = ax4.imshow(corr_true, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax4.set_title('D) Correlation - True Data')
    ax4.set_xlabel('Biomarker')
    ax4.set_ylabel('Biomarker')
    plt.colorbar(im, ax=ax4)

    # Panel E: Correlation matrix - Observed
    ax5 = fig.add_subplot(gs[1, 1])
    corr_obs = np.corrcoef(X_obs.T)
    im2 = ax5.imshow(corr_obs, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax5.set_title('E) Correlation - Observed Data')
    ax5.set_xlabel('Biomarker')
    ax5.set_ylabel('Biomarker')
    plt.colorbar(im2, ax=ax5)

    # Panel F: Distribution shift
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(X_true[:, 0], bins=30, alpha=0.5, color='blue', label='True', density=True)
    ax6.hist(X_obs[:, 0], bins=30, alpha=0.5, color='red', label='Observed', density=True)
    ax6.set_xlabel('Concentration (Biomarker 1)')
    ax6.set_ylabel('Density')
    ax6.set_title('F) Distribution Shift')
    ax6.legend()

    fig.suptitle('Figure 2: Impact of Dilution on Biomarker Data', fontsize=14,
                 fontweight='bold', y=1.02)
    fig.savefig(FIGURES_DIR / 'fig2_dilution_impact.png')
    fig.savefig(FIGURES_DIR / 'fig2_dilution_impact.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: fig2_dilution_impact.png")
    return dataset


def analysis_3_normalization_comparison():
    """Compare normalization methods across dilution severities (Figure 3, Table 2)."""
    print("\n" + "=" * 60)
    print("Analysis 3: Normalization Method Comparison")
    print("=" * 60)

    np.random.seed(42)

    norm_methods = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile', 'reference']
    dilution_configs = {
        'Mild': {'dilution_alpha': 8.0, 'dilution_beta': 2.0},
        'Moderate': {'dilution_alpha': 5.0, 'dilution_beta': 5.0},
        'Severe': {'dilution_alpha': 2.0, 'dilution_beta': 8.0},
    }

    n_replications = 20
    all_results = []

    for severity, dil_params in dilution_configs.items():
        print(f"  Running {severity} dilution ({n_replications} replications)...")
        for rep in range(n_replications):
            np.random.seed(42 + rep)
            params = {
                'n_subjects': 100,
                'n_biomarkers': 10,
                'n_groups': 2,
                'correlation_type': 'moderate',
                'effect_size': 0.8,
                'distribution': 'lognormal',
                'lod_percentile': 0.1,
                'lod_handling': 'substitute',
                **dil_params,
            }

            try:
                results = run_single_simulation(params, norm_methods)

                for method in norm_methods:
                    row = {
                        'severity': severity,
                        'replication': rep,
                        'method': method,
                        'power': results['univariate'][method]['power'],
                        'type_i_error': results['univariate'][method]['type_i_error'],
                        'corr_frobenius': results['correlation'][method]['frobenius_norm'],
                        'corr_rank': results['correlation'][method]['rank_correlation'],
                        'pca_rv': results['pca'][method]['rv_coefficient'],
                        'clustering_ari': results['clustering'][method]['adjusted_rand_index'],
                        'classification_acc': results['classification'][method]['accuracy'],
                        'classification_auc': results['classification'][method].get('auc_roc', np.nan),
                    }
                    all_results.append(row)
            except Exception as e:
                print(f"    Warning: replication {rep} failed for {severity}: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'normalization_results.csv', index=False)

    # Figure 3: Normalization performance by severity
    metrics = {
        'power': 'Statistical Power',
        'type_i_error': 'Type I Error Rate',
        'corr_rank': 'Correlation Recovery',
        'pca_rv': 'PCA Structure (RV Coeff.)',
        'clustering_ari': 'Clustering (Adj. Rand Index)',
        'classification_acc': 'Classification Accuracy',
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, (metric, title) in enumerate(metrics.items()):
        ax = axes[idx // 3, idx % 3]
        sns.boxplot(data=df, x='method', y=metric, hue='severity', ax=ax,
                    palette={'Mild': '#2196F3', 'Moderate': '#FF9800', 'Severe': '#F44336'},
                    fliersize=2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Normalization Method')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=45)
        if idx == 0:
            ax.legend(title='Dilution', fontsize=8)
        else:
            ax.get_legend().remove()

        # Reference line for Type I error
        if metric == 'type_i_error':
            ax.axhline(0.05, color='black', linestyle='--', linewidth=1, alpha=0.5)

    fig.suptitle('Figure 3: Normalization Method Performance Across Dilution Severities',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_normalization_comparison.png')
    fig.savefig(FIGURES_DIR / 'fig3_normalization_comparison.pdf', format='pdf')
    plt.close(fig)

    # Table 2: Summary statistics
    summary = df.groupby(['severity', 'method']).agg(
        power_mean=('power', 'mean'),
        power_std=('power', 'std'),
        type_i_mean=('type_i_error', 'mean'),
        corr_rank_mean=('corr_rank', 'mean'),
        pca_rv_mean=('pca_rv', 'mean'),
        clustering_ari_mean=('clustering_ari', 'mean'),
        class_acc_mean=('classification_acc', 'mean'),
    ).round(3).reset_index()
    summary.to_csv(RESULTS_DIR / 'table2_normalization_summary.csv', index=False)
    print(f"  Saved: fig3_normalization_comparison.png, table2_normalization_summary.csv")
    print(f"  Results: {len(df)} rows across {df['severity'].nunique()} severities")
    return df


def analysis_4_sample_size_effect():
    """Evaluate impact of sample size and effect size (Figure 4)."""
    print("\n" + "=" * 60)
    print("Analysis 4: Sample Size and Effect Size Interaction")
    print("=" * 60)

    np.random.seed(42)

    sample_sizes = [30, 50, 100, 200]
    effect_sizes = [0.2, 0.5, 0.8, 1.2]
    norm_methods = ['none', 'total_sum', 'pqn', 'clr']
    n_reps = 10

    rows = []
    total = len(sample_sizes) * len(effect_sizes) * n_reps
    count = 0
    for n_sub in sample_sizes:
        for es in effect_sizes:
            for rep in range(n_reps):
                np.random.seed(100 + rep)
                count += 1
                if count % 20 == 0:
                    print(f"  Progress: {count}/{total}")
                params = {
                    'n_subjects': n_sub,
                    'n_biomarkers': 10,
                    'n_groups': 2,
                    'correlation_type': 'moderate',
                    'effect_size': es,
                    'distribution': 'lognormal',
                    'dilution_alpha': 3.0,
                    'dilution_beta': 5.0,
                    'lod_percentile': 0.1,
                    'lod_handling': 'substitute',
                }
                try:
                    results = run_single_simulation(params, norm_methods)
                    for method in norm_methods:
                        rows.append({
                            'n_subjects': n_sub,
                            'effect_size': es,
                            'rep': rep,
                            'method': method,
                            'power': results['univariate'][method]['power'],
                            'classification_acc': results['classification'][method]['accuracy'],
                        })
                except Exception as e:
                    pass

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'sample_effect_results.csv', index=False)

    # Figure 4: Heatmap of power by sample size x effect size for each method
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, method in enumerate(norm_methods):
        ax = axes[idx // 2, idx % 2]
        subset = df[df['method'] == method]
        pivot = subset.pivot_table(index='n_subjects', columns='effect_size',
                                   values='power', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                    vmin=0, vmax=1, cbar_kws={'label': 'Power'})
        ax.set_title(f'{method.upper()}', fontweight='bold')
        ax.set_xlabel('Effect Size')
        ax.set_ylabel('Sample Size')

    fig.suptitle('Figure 4: Statistical Power by Sample Size and Effect Size',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_sample_effect_power.png')
    fig.savefig(FIGURES_DIR / 'fig4_sample_effect_power.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: fig4_sample_effect_power.png")
    return df


def analysis_5_ml_robustness():
    """Evaluate ML method robustness to dilution (Figure 5)."""
    print("\n" + "=" * 60)
    print("Analysis 5: Machine Learning Robustness to Dilution")
    print("=" * 60)

    np.random.seed(42)

    ml_methods = ['logistic', 'random_forest', 'gradient_boosting']
    dilution_configs = {
        'No Dilution': None,
        'Mild': {'dilution_alpha': 8.0, 'dilution_beta': 2.0},
        'Moderate': {'dilution_alpha': 5.0, 'dilution_beta': 5.0},
        'Severe': {'dilution_alpha': 2.0, 'dilution_beta': 8.0},
    }
    norm_methods_ml = ['none', 'pqn', 'clr']
    n_reps = 10
    rows = []

    for severity, dil_params in dilution_configs.items():
        print(f"  Running {severity}...")
        for rep in range(n_reps):
            np.random.seed(200 + rep)
            params = {
                'n_subjects': 150,
                'n_biomarkers': 15,
                'n_groups': 2,
                'correlation_type': 'moderate',
                'effect_size': 0.8,
                'distribution': 'lognormal',
                'lod_percentile': 0.1,
                'lod_handling': 'substitute',
            }
            if dil_params:
                params.update(dil_params)
            else:
                params['dilution_alpha'] = 100.0
                params['dilution_beta'] = 1.0

            try:
                dataset = generate_dataset(**params)
                X_obs = dataset['X_obs']
                y = dataset['y']

                for norm in norm_methods_ml:
                    if norm == 'none':
                        X_use = X_obs.copy()
                    else:
                        X_use = normalize_data(X_obs, method=norm)

                    for ml_method in ml_methods:
                        try:
                            cv_results = cross_validate_classification(
                                X_use, y, ml_method, n_folds=5
                            )
                            summary = cv_results.get('summary', {})
                            rows.append({
                                'severity': severity,
                                'rep': rep,
                                'normalization': norm,
                                'ml_method': ml_method,
                                'accuracy': summary.get('accuracy_mean', np.nan),
                                'auc': summary.get('auc_roc_mean', np.nan),
                            })
                        except Exception:
                            pass
            except Exception as e:
                print(f"    Warning: rep {rep} failed for {severity}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'ml_robustness_results.csv', index=False)

    # Figure 5: ML accuracy by dilution severity and normalization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, ml_method in enumerate(ml_methods):
        ax = axes[idx]
        subset = df[df['ml_method'] == ml_method]
        sns.barplot(data=subset, x='severity', y='accuracy', hue='normalization',
                    ax=ax, palette='Set2', ci=95,
                    order=['No Dilution', 'Mild', 'Moderate', 'Severe'])
        ax.set_title(f'{ml_method.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Dilution Severity')
        ax.set_ylabel('Classification Accuracy')
        ax.set_ylim(0.4, 1.0)
        ax.tick_params(axis='x', rotation=30)
        legend = ax.get_legend()
        if idx > 0 and legend is not None:
            legend.remove()
        elif idx == 0 and legend is not None:
            legend.set_title('Normalization')

    fig.suptitle('Figure 5: ML Classification Accuracy Under Different Dilution Conditions',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig5_ml_robustness.png')
    fig.savefig(FIGURES_DIR / 'fig5_ml_robustness.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: fig5_ml_robustness.png")
    return df


def analysis_6_feature_selection():
    """Evaluate feature selection stability under dilution (Figure 6)."""
    print("\n" + "=" * 60)
    print("Analysis 6: Feature Selection Robustness")
    print("=" * 60)

    np.random.seed(42)

    n_biomarkers = 20
    n_features_select = 8
    fs_methods = ['lasso', 'random_forest', 'mutual_information']
    dilution_configs = {
        'Mild': {'dilution_alpha': 8.0, 'dilution_beta': 2.0},
        'Moderate': {'dilution_alpha': 5.0, 'dilution_beta': 5.0},
        'Severe': {'dilution_alpha': 2.0, 'dilution_beta': 8.0},
    }
    n_reps = 15
    rows = []

    for severity, dil_params in dilution_configs.items():
        print(f"  Running {severity}...")
        for rep in range(n_reps):
            np.random.seed(300 + rep)
            params = {
                'n_subjects': 150,
                'n_biomarkers': n_biomarkers,
                'n_groups': 2,
                'correlation_type': 'moderate',
                'effect_size': 0.8,
                'distribution': 'lognormal',
                'lod_percentile': 0.1,
                'lod_handling': 'substitute',
                **dil_params,
            }
            dataset = generate_dataset(**params)
            X_true = dataset['X_true']
            X_obs = dataset['X_obs']
            y = dataset['y']

            # Get "ground truth" features from true data
            try:
                true_features, _ = feature_selection(X_true, y, method='random_forest',
                                                     n_features=n_features_select)
                true_set = set(true_features)
            except Exception:
                continue

            for fs_method in fs_methods:
                for norm in ['none', 'pqn', 'clr']:
                    try:
                        if norm == 'none':
                            X_use = X_obs.copy()
                        else:
                            X_use = normalize_data(X_obs, method=norm)

                        sel_features, importances = feature_selection(
                            X_use, y, method=fs_method, n_features=n_features_select
                        )
                        sel_set = set(sel_features)

                        # Jaccard similarity with true features
                        if len(true_set | sel_set) > 0:
                            jaccard = len(true_set & sel_set) / len(true_set | sel_set)
                        else:
                            jaccard = 0.0

                        overlap = len(true_set & sel_set) / len(true_set)

                        rows.append({
                            'severity': severity,
                            'rep': rep,
                            'fs_method': fs_method,
                            'normalization': norm,
                            'jaccard': jaccard,
                            'overlap': overlap,
                            'n_correct': len(true_set & sel_set),
                        })
                    except Exception:
                        pass

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'feature_selection_results.csv', index=False)

    # Figure 6: Feature selection robustness
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, fs_method in enumerate(fs_methods):
        ax = axes[idx]
        subset = df[df['fs_method'] == fs_method]
        if subset.empty:
            ax.set_title(f'{fs_method.replace("_", " ").title()} (no data)', fontweight='bold')
            continue
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        norm_labels = subset['normalization'].unique()
        palette = {n: colors[i % len(colors)] for i, n in enumerate(sorted(norm_labels))}
        sns.boxplot(data=subset, x='severity', y='jaccard', hue='normalization',
                    ax=ax, palette=palette, order=['Mild', 'Moderate', 'Severe'],
                    fliersize=2)
        ax.set_title(f'{fs_method.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Dilution Severity')
        ax.set_ylabel('Jaccard Similarity with True Features')
        ax.set_ylim(0, 1.05)
        legend = ax.get_legend()
        if idx > 0 and legend is not None:
            legend.remove()
        elif idx == 0 and legend is not None:
            legend.set_title('Normalization')

    fig.suptitle('Figure 6: Feature Selection Robustness Under Dilution',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig6_feature_selection.png')
    fig.savefig(FIGURES_DIR / 'fig6_feature_selection.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: fig6_feature_selection.png")
    return df


def analysis_7_power_analysis():
    """Power analysis under dilution (Figure 7)."""
    print("\n" + "=" * 60)
    print("Analysis 7: Power Analysis Under Dilution")
    print("=" * 60)

    np.random.seed(42)

    # Theoretical power curves
    effect_sizes_arr = np.array([0.2, 0.5, 0.8, 1.0, 1.5])
    sample_sizes_arr = np.arange(10, 201, 10)

    fig = plot_power_curve(effect_sizes_arr, sample_sizes_arr)
    fig.savefig(FIGURES_DIR / 'fig7a_power_curves_theoretical.png')
    fig.savefig(FIGURES_DIR / 'fig7a_power_curves_theoretical.pdf', format='pdf')
    plt.close(fig)

    # Sample size recommendations
    recommendations = []
    for es in [0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
        n_req = sample_size_estimation(es, power=0.8)
        recommendations.append({'effect_size': es, 'n_per_group_80pct_power': n_req})

    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(RESULTS_DIR / 'table3_sample_size_recommendations.csv', index=False)

    # Empirical power under dilution
    sample_sizes_test = [30, 50, 100, 150, 200]
    dilution_configs = {
        'None': {'dilution_alpha': 100.0, 'dilution_beta': 1.0},
        'Mild': {'dilution_alpha': 8.0, 'dilution_beta': 2.0},
        'Severe': {'dilution_alpha': 2.0, 'dilution_beta': 8.0},
    }
    n_reps = 10
    rows = []

    for severity, dil_params in dilution_configs.items():
        for n_sub in sample_sizes_test:
            for rep in range(n_reps):
                np.random.seed(400 + rep)
                params = {
                    'n_subjects': n_sub,
                    'n_biomarkers': 10,
                    'n_groups': 2,
                    'correlation_type': 'moderate',
                    'effect_size': 0.8,
                    'distribution': 'lognormal',
                    'lod_percentile': 0.1,
                    'lod_handling': 'substitute',
                    **dil_params,
                }
                try:
                    results = run_single_simulation(params, ['none', 'pqn', 'clr'])
                    for method in ['none', 'pqn', 'clr']:
                        rows.append({
                            'severity': severity,
                            'n_subjects': n_sub,
                            'rep': rep,
                            'method': method,
                            'power': results['univariate'][method]['power'],
                        })
                except Exception:
                    pass

    power_df = pd.DataFrame(rows)
    power_df.to_csv(RESULTS_DIR / 'empirical_power_results.csv', index=False)

    # Figure 7b: Empirical power curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, method in enumerate(['none', 'pqn', 'clr']):
        ax = axes[idx]
        subset = power_df[power_df['method'] == method]
        for severity in ['None', 'Mild', 'Severe']:
            sev_data = subset[subset['severity'] == severity]
            means = sev_data.groupby('n_subjects')['power'].mean()
            sems = sev_data.groupby('n_subjects')['power'].sem()
            ax.errorbar(means.index, means.values, yerr=sems.values,
                        marker='o', capsize=3, label=severity, linewidth=2)
        ax.set_title(f'Normalization: {method.upper()}', fontweight='bold')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Empirical Power')
        ax.set_ylim(0, 1.05)
        ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='80% Power')
        ax.legend()

    fig.suptitle('Figure 7: Empirical Power Under Dilution',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig7b_empirical_power.png')
    fig.savefig(FIGURES_DIR / 'fig7b_empirical_power.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: fig7a, fig7b, table3_sample_size_recommendations.csv")
    print(f"  Sample size recommendations:")
    print(rec_df.to_string(index=False))
    return power_df


def analysis_8_enhanced_statistical():
    """Enhanced statistical analysis with effect sizes and CIs (Figure 8)."""
    print("\n" + "=" * 60)
    print("Analysis 8: Enhanced Statistical Analysis")
    print("=" * 60)

    np.random.seed(42)

    params = {
        'n_subjects': 200,
        'n_biomarkers': 15,
        'n_groups': 2,
        'correlation_type': 'moderate',
        'effect_size': 0.8,
        'distribution': 'lognormal',
        'dilution_alpha': 3.0,
        'dilution_beta': 5.0,
        'lod_percentile': 0.1,
        'lod_handling': 'substitute',
    }

    dataset = generate_dataset(**params)
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    y = dataset['y']

    # Effect sizes: true vs observed
    es_true = calculate_effect_size(X_true, y, 'cohens_d')
    es_obs = calculate_effect_size(X_obs, y, 'cohens_d')

    # Bootstrap CIs for observed
    es_point, es_lower, es_upper = bootstrap_confidence_interval(
        X_obs, y, lambda x, yy: calculate_effect_size(x, yy, 'cohens_d'),
        n_bootstrap=200
    )

    # Enhanced univariate
    enhanced = analyze_univariate_enhanced(X_obs, y)

    # Volcano plot
    if 'fold_changes' in enhanced and 'p_values' in enhanced:
        fig_volcano = plot_volcano(
            enhanced['fold_changes'], enhanced['p_values'],
            title="Volcano Plot - Observed Data Under Dilution"
        )
        fig_volcano.savefig(FIGURES_DIR / 'fig8a_volcano.png')
        plt.close(fig_volcano)

    # Forest plot
    fig_forest = plot_forest(es_obs, es_lower, es_upper)
    fig_forest.savefig(FIGURES_DIR / 'fig8b_forest.png')
    plt.close(fig_forest)

    # Figure 8c: Effect size comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    biomarker_ids = np.arange(len(es_true))
    width = 0.35
    ax.bar(biomarker_ids - width / 2, np.abs(es_true), width, label='True', color='#2196F3', alpha=0.8)
    ax.bar(biomarker_ids + width / 2, np.abs(es_obs), width, label='Observed', color='#F44336', alpha=0.8)
    ax.set_xlabel('Biomarker')
    ax.set_ylabel("|Cohen's d|")
    ax.set_title("Figure 8c: Effect Size Attenuation Due to Dilution", fontweight='bold')
    ax.legend()
    ax.set_xticks(biomarker_ids)
    ax.set_xticklabels([f'B{i + 1}' for i in biomarker_ids])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig8c_effect_size_comparison.png')
    fig.savefig(FIGURES_DIR / 'fig8c_effect_size_comparison.pdf', format='pdf')
    plt.close(fig)

    # Multiple testing
    p_values, _ = analyze_univariate(X_obs, y, 't_test')
    p_fdr, sig_fdr = multiple_testing_correction(p_values, 'fdr_bh')
    p_bonf, sig_bonf = multiple_testing_correction(p_values, 'bonferroni')

    stats_summary = {
        'n_biomarkers': len(es_true),
        'mean_abs_true_es': float(np.mean(np.abs(es_true))),
        'mean_abs_obs_es': float(np.mean(np.abs(es_obs))),
        'es_attenuation_pct': float((1 - np.mean(np.abs(es_obs)) / np.mean(np.abs(es_true))) * 100),
        'n_significant_fdr': int(np.sum(sig_fdr)),
        'n_significant_bonferroni': int(np.sum(sig_bonf)),
        'mean_ci_width': float(np.mean(es_upper - es_lower)),
    }

    with open(RESULTS_DIR / 'enhanced_stats_summary.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)

    print(f"  Effect size attenuation: {stats_summary['es_attenuation_pct']:.1f}%")
    print(f"  Significant (FDR): {stats_summary['n_significant_fdr']}/{len(es_true)}")
    print(f"  Significant (Bonferroni): {stats_summary['n_significant_bonferroni']}/{len(es_true)}")
    print(f"  Saved: fig8a_volcano, fig8b_forest, fig8c_effect_size_comparison")
    return stats_summary


def analysis_9_high_dimensional():
    """High-dimensional extension: p in {50, 100, 500} biomarkers (Figure E1)."""
    print("\n" + "=" * 60)
    print("Analysis 9: High-Dimensional Extension")
    print("=" * 60)

    np.random.seed(42)

    p_values_list = [10, 50, 100, 500]
    norm_methods = ['none', 'clr', 'pqn', 'quantile']
    n_reps = 10
    rows = []

    for p in p_values_list:
        print(f"  Testing p={p} biomarkers...")
        for rep in range(n_reps):
            np.random.seed(500 + rep)
            t_start = time.time()
            params = {
                'n_subjects': 100,
                'n_biomarkers': p,
                'n_groups': 2,
                'correlation_type': 'block',
                'block_size': min(5, p),
                'effect_size': 0.8,
                'distribution': 'lognormal',
                'dilution_alpha': 5.0,
                'dilution_beta': 5.0,
                'lod_percentile': 0.1,
                'lod_handling': 'substitute',
            }
            try:
                results = run_single_simulation(params, norm_methods)
                comp_time = time.time() - t_start

                for method in norm_methods:
                    power_raw = results['univariate'][method]['power']
                    type1_raw = results['univariate'][method]['type_i_error']

                    # Also compute FDR-corrected metrics
                    dataset = generate_dataset(**params)
                    X_obs = dataset['X_obs']
                    y = dataset['y']
                    if method == 'none':
                        X_use = X_obs.copy()
                    else:
                        X_use = normalize_data(X_obs, method=method)

                    p_vals, _ = analyze_univariate(X_use, y, 't_test')
                    p_fdr, sig_fdr = multiple_testing_correction(p_vals, 'fdr_bh')

                    # Determine which are truly differential (~70%)
                    n_differential = int(p * 0.7)
                    is_differential = np.zeros(p, dtype=bool)
                    is_differential[:n_differential] = True

                    power_fdr = np.mean(sig_fdr[is_differential]) if n_differential > 0 else 0.0
                    type1_fdr = np.mean(sig_fdr[~is_differential]) if (p - n_differential) > 0 else 0.0

                    rows.append({
                        'p': p,
                        'rep': rep,
                        'method': method,
                        'power_raw': power_raw,
                        'type1_raw': type1_raw,
                        'power_fdr': power_fdr,
                        'type1_fdr': type1_fdr,
                        'comp_time': comp_time,
                        'clustering_ari': results['clustering'][method]['adjusted_rand_index'],
                        'classification_acc': results['classification'][method]['accuracy'],
                    })
            except Exception as e:
                print(f"    Warning: p={p}, rep={rep} failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'high_dimensional_results.csv', index=False)

    # Figure E1: High-dimensional performance
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Power with and without FDR
    ax = axes[0]
    for method in norm_methods:
        subset = df[df['method'] == method]
        means = subset.groupby('p')['power_raw'].mean()
        ax.plot(means.index, means.values, 'o-', label=f'{method} (raw)', linewidth=2)
        means_fdr = subset.groupby('p')['power_fdr'].mean()
        ax.plot(means_fdr.index, means_fdr.values, 's--', label=f'{method} (FDR)', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Number of Biomarkers (p)')
    ax.set_ylabel('Statistical Power')
    ax.set_title('A) Power vs. Dimensionality', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xscale('log')

    # Panel B: Type I error with and without FDR
    ax = axes[1]
    for method in norm_methods:
        subset = df[df['method'] == method]
        means = subset.groupby('p')['type1_raw'].mean()
        ax.plot(means.index, means.values, 'o-', label=f'{method} (raw)', linewidth=2)
        means_fdr = subset.groupby('p')['type1_fdr'].mean()
        ax.plot(means_fdr.index, means_fdr.values, 's--', label=f'{method} (FDR)', linewidth=1.5, alpha=0.7)
    ax.axhline(0.05, color='black', linestyle=':', alpha=0.5, label='Nominal α')
    ax.set_xlabel('Number of Biomarkers (p)')
    ax.set_ylabel('Type I Error Rate')
    ax.set_title('B) Type I Error vs. Dimensionality', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xscale('log')

    # Panel C: Computation time
    ax = axes[2]
    means = df.groupby('p')['comp_time'].mean()
    ax.plot(means.index, means.values, 'ko-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Biomarkers (p)')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('C) Computation Time Scaling', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.suptitle('Figure E1: High-Dimensional Extension', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE1_high_dimensional.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE1_high_dimensional.png')
    plt.close(fig)
    print(f"  Saved: figE1_high_dimensional.pdf, high_dimensional_results.csv")
    return df


def analysis_10_type1_error_decomposition():
    """CLR Type I error decomposition: standard t-test vs permutation (Figure E2)."""
    print("\n" + "=" * 60)
    print("Analysis 10: CLR Type I Error Decomposition")
    print("=" * 60)

    np.random.seed(42)

    norm_methods = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile']
    n_reps = 20
    n_permutations = 1000  # per biomarker
    p = 10
    n = 100

    rows = []
    all_pvalues = {method: [] for method in norm_methods}

    for rep in range(n_reps):
        np.random.seed(600 + rep)
        if (rep + 1) % 5 == 0:
            print(f"  Replication {rep + 1}/{n_reps}")

        # Generate NULL dataset: zero effect size for ALL biomarkers
        params = {
            'n_subjects': n,
            'n_biomarkers': p,
            'n_groups': 2,
            'correlation_type': 'moderate',
            'effect_size': 0.0,  # NULL - no true differences
            'distribution': 'lognormal',
            'dilution_alpha': 5.0,
            'dilution_beta': 5.0,
            'lod_percentile': 0.1,
            'lod_handling': 'substitute',
        }
        dataset = generate_dataset(**params)
        X_obs = dataset['X_obs']
        y = dataset['y']

        for method in norm_methods:
            if method == 'none':
                X_use = X_obs.copy()
            else:
                X_use = normalize_data(X_obs, method=method)

            # Standard t-test
            p_vals_ttest, _ = analyze_univariate(X_use, y, 't_test')
            type1_ttest = np.mean(p_vals_ttest < 0.05)
            all_pvalues[method].extend(p_vals_ttest.tolist())

            # Permutation test
            perm_pvals = np.zeros(p)
            for j in range(p):
                # Observed test statistic
                g0 = X_use[y == 0, j]
                g1 = X_use[y == 1, j]
                obs_stat = abs(np.mean(g1) - np.mean(g0))

                # Permutation distribution
                count_extreme = 0
                combined = np.concatenate([g0, g1])
                n0 = len(g0)
                for _ in range(n_permutations):
                    perm = np.random.permutation(combined)
                    perm_stat = abs(np.mean(perm[n0:]) - np.mean(perm[:n0]))
                    if perm_stat >= obs_stat:
                        count_extreme += 1
                perm_pvals[j] = (count_extreme + 1) / (n_permutations + 1)

            type1_perm = np.mean(perm_pvals < 0.05)

            rows.append({
                'rep': rep,
                'method': method,
                'type1_ttest': type1_ttest,
                'type1_perm': type1_perm,
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'type1_error_decomposition.csv', index=False)

    # Figure E2: Type I error decomposition
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Bar chart comparing t-test vs permutation
    ax = axes[0]
    summary = df.groupby('method').agg(
        ttest_mean=('type1_ttest', 'mean'),
        ttest_sem=('type1_ttest', 'sem'),
        perm_mean=('type1_perm', 'mean'),
        perm_sem=('type1_perm', 'sem'),
    ).reindex(norm_methods)

    x = np.arange(len(norm_methods))
    width = 0.35
    ax.bar(x - width / 2, summary['ttest_mean'], width, yerr=summary['ttest_sem'],
           label='Standard t-test', color='#F44336', alpha=0.8, capsize=3)
    ax.bar(x + width / 2, summary['perm_mean'], width, yerr=summary['perm_sem'],
           label='Permutation test', color='#2196F3', alpha=0.8, capsize=3)
    ax.axhline(0.05, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Nominal α = 0.05')
    ax.set_xlabel('Normalization Method')
    ax.set_ylabel('Type I Error Rate')
    ax.set_title('A) Type I Error Under Null (Zero Effect Size)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(norm_methods, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)

    # Panel B: QQ-plot for CLR p-values
    ax = axes[1]
    clr_pvals = np.array(all_pvalues['clr'])
    none_pvals = np.array(all_pvalues['none'])

    # Sort and compute expected uniform quantiles
    n_total = len(clr_pvals)
    expected = np.arange(1, n_total + 1) / (n_total + 1)

    clr_sorted = np.sort(clr_pvals)
    none_sorted = np.sort(none_pvals)

    ax.plot(-np.log10(expected), -np.log10(none_sorted), 'o', color='#4CAF50',
            alpha=0.5, markersize=3, label='None (baseline)')
    ax.plot(-np.log10(expected), -np.log10(clr_sorted), 'o', color='#F44336',
            alpha=0.5, markersize=3, label='CLR')
    max_val = max(-np.log10(expected).max(), -np.log10(clr_sorted[clr_sorted > 0]).max()) if any(clr_sorted > 0) else 5
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Expected (uniform)')
    ax.set_xlabel('Expected -log₁₀(p)')
    ax.set_ylabel('Observed -log₁₀(p)')
    ax.set_title('B) QQ-Plot of P-values Under Null', fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle('Figure E2: CLR Type I Error Decomposition', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE2_type1_decomposition.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE2_type1_decomposition.png')
    plt.close(fig)
    print(f"  Saved: figE2_type1_decomposition.pdf, type1_error_decomposition.csv")

    # Print summary
    summary_print = df.groupby('method')[['type1_ttest', 'type1_perm']].mean()
    print("  Type I Error Summary (mean across replications):")
    print(summary_print.to_string())
    return df


def analysis_11_multigroup():
    """Multi-group comparison extension: 2, 3, 4 groups (Figure E5)."""
    print("\n" + "=" * 60)
    print("Analysis 11: Multi-Group Comparison")
    print("=" * 60)

    np.random.seed(42)

    n_groups_list = [2, 3, 4]
    norm_methods = ['none', 'clr', 'pqn', 'quantile']
    n_reps = 10
    rows = []

    for ng in n_groups_list:
        print(f"  Testing n_groups={ng}...")
        for rep in range(n_reps):
            np.random.seed(700 + rep)
            params = {
                'n_subjects': 150,
                'n_biomarkers': 10,
                'n_groups': ng,
                'correlation_type': 'moderate',
                'effect_size': 0.8,
                'distribution': 'lognormal',
                'dilution_alpha': 5.0,
                'dilution_beta': 5.0,
                'lod_percentile': 0.1,
                'lod_handling': 'substitute',
            }
            try:
                results = run_single_simulation(params, norm_methods)
                for method in norm_methods:
                    rows.append({
                        'n_groups': ng,
                        'rep': rep,
                        'method': method,
                        'power': results['univariate'][method]['power'],
                        'type_i_error': results['univariate'][method]['type_i_error'],
                        'clustering_ari': results['clustering'][method]['adjusted_rand_index'],
                        'classification_acc': results['classification'][method]['accuracy'],
                    })
            except Exception as e:
                print(f"    Warning: n_groups={ng}, rep={rep} failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'multigroup_results.csv', index=False)

    # Figure E5: Multi-group analysis
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Power by number of groups
    ax = axes[0]
    sns.boxplot(data=df, x='n_groups', y='power', hue='method', ax=ax, palette='Set2', fliersize=2)
    ax.set_title('A) Power by Number of Groups', fontweight='bold')
    ax.set_xlabel('Number of Groups')
    ax.set_ylabel('Statistical Power')
    ax.legend(title='Normalization', fontsize=8)

    # Panel B: Clustering ARI
    ax = axes[1]
    sns.boxplot(data=df, x='n_groups', y='clustering_ari', hue='method', ax=ax, palette='Set2', fliersize=2)
    ax.set_title('B) Clustering ARI by Number of Groups', fontweight='bold')
    ax.set_xlabel('Number of Groups')
    ax.set_ylabel('Adjusted Rand Index')
    legend = ax.get_legend()
    if legend:
        legend.remove()

    # Panel C: Classification accuracy
    ax = axes[2]
    sns.boxplot(data=df, x='n_groups', y='classification_acc', hue='method', ax=ax, palette='Set2', fliersize=2)
    ax.set_title('C) Classification Accuracy by Groups', fontweight='bold')
    ax.set_xlabel('Number of Groups')
    ax.set_ylabel('Accuracy')
    legend = ax.get_legend()
    if legend:
        legend.remove()

    fig.suptitle('Figure E5: Multi-Group Extension', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE5_multigroup.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE5_multigroup.png')
    plt.close(fig)
    print(f"  Saved: figE5_multigroup.pdf, multigroup_results.csv")
    return df


def analysis_12_batch_dilution_interaction():
    """Batch effects + dilution interaction (Figure E3)."""
    print("\n" + "=" * 60)
    print("Analysis 12: Batch-Dilution Interaction")
    print("=" * 60)

    np.random.seed(42)

    n_reps = 15
    rows = []

    for rep in range(n_reps):
        np.random.seed(800 + rep)
        if (rep + 1) % 5 == 0:
            print(f"  Replication {rep + 1}/{n_reps}")

        # Generate data with both batch effects and dilution
        params = {
            'n_subjects': 150,
            'n_biomarkers': 10,
            'n_groups': 2,
            'correlation_type': 'moderate',
            'effect_size': 0.8,
            'distribution': 'lognormal',
            'dilution_alpha': 5.0,
            'dilution_beta': 5.0,
            'lod_percentile': 0.1,
            'lod_handling': 'substitute',
        }

        dataset = generate_dataset(**params)
        X_true = dataset['X_true']
        X_obs = dataset['X_obs']
        y = dataset['y']

        # Add batch effects
        n_batches = 3
        batch_assignments, batch_multipliers = simulate_batch_effects(
            params['n_subjects'], n_batches, batch_variability=0.3
        )
        X_obs_batch = X_obs * batch_multipliers.reshape(-1, 1)

        # Correction strategies
        strategies = {}

        # 1. No correction
        strategies['None'] = X_obs_batch.copy()

        # 2. Dilution only (CLR)
        strategies['Dilution only (CLR)'] = normalize_data(X_obs_batch, method='clr')

        # 3. Batch only (ComBat)
        try:
            X_combat = normalize_combat(X_obs_batch, batch_assignments)
            strategies['Batch only (ComBat)'] = X_combat
        except Exception:
            strategies['Batch only (ComBat)'] = X_obs_batch.copy()

        # 4. Combined: ComBat then CLR
        try:
            X_combat_clr = normalize_data(X_combat, method='clr')
            strategies['Combined (ComBat+CLR)'] = X_combat_clr
        except Exception:
            strategies['Combined (ComBat+CLR)'] = X_obs_batch.copy()

        # Get true p-values for evaluation
        p_vals_true, _ = analyze_univariate(X_true, y, 't_test')

        for strat_name, X_corrected in strategies.items():
            try:
                # Univariate: compare to true p-values
                p_vals_obs, _ = analyze_univariate(X_corrected, y, 't_test')
                univ_results = evaluate_univariate(p_vals_true, p_vals_obs, 0.05)

                # Clustering
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = km.fit_predict(X_corrected)
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(y, labels)

                # Classification
                from sklearn.model_selection import cross_val_score
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=1000, random_state=42)
                scores = cross_val_score(clf, X_corrected, y, cv=5, scoring='accuracy')

                rows.append({
                    'rep': rep,
                    'strategy': strat_name,
                    'power': univ_results.get('power', np.nan),
                    'type_i_error': univ_results.get('type_i_error', np.nan),
                    'clustering_ari': ari,
                    'classification_acc': np.mean(scores),
                })
            except Exception as e:
                rows.append({
                    'rep': rep,
                    'strategy': strat_name,
                    'power': np.nan,
                    'type_i_error': np.nan,
                    'clustering_ari': np.nan,
                    'classification_acc': np.nan,
                })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'batch_dilution_results.csv', index=False)

    # Figure E3: Batch-dilution interaction
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    strategy_order = ['None', 'Dilution only (CLR)', 'Batch only (ComBat)', 'Combined (ComBat+CLR)']

    metrics_plot = {
        'classification_acc': ('A) Classification Accuracy', axes[0, 0]),
        'clustering_ari': ('B) Clustering ARI', axes[0, 1]),
        'power': ('C) Statistical Power', axes[1, 0]),
        'type_i_error': ('D) Type I Error', axes[1, 1]),
    }

    colors_strat = ['#F44336', '#FF9800', '#2196F3', '#4CAF50']

    for metric, (title, ax) in metrics_plot.items():
        sns.boxplot(data=df, x='strategy', y=metric, ax=ax, palette=colors_strat,
                    order=strategy_order, fliersize=2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
        if metric == 'type_i_error':
            ax.axhline(0.05, color='black', linestyle='--', alpha=0.5)

    fig.suptitle('Figure E3: Batch-Dilution Interaction', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE3_batch_dilution.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE3_batch_dilution.png')
    plt.close(fig)
    print(f"  Saved: figE3_batch_dilution.pdf, batch_dilution_results.csv")
    return df


def analysis_13_lod_sensitivity():
    """Evaluate LOD severity × imputation method interaction (Figure E6, Table E5)."""
    print("\n" + "=" * 60)
    print("Analysis 13: LOD Sensitivity — Severity × Imputation Method")
    print("=" * 60)

    np.random.seed(42)

    # LOD severity: fraction of values below LOD (controlled via percentile of
    # the *diluted* distribution used to set the LOD threshold)
    lod_severities = {
        '5%':  0.05,
        '10%': 0.10,
        '20%': 0.20,
        '30%': 0.30,
        '50%': 0.50,
    }

    # Imputation methods
    imputation_methods = {
        'Zero':       'zero',
        'LOD/2':      'lod_half',
        'LOD/√2':     'substitute',  # Hornung & Reed 1990
        'LOD':        'lod',
        'MICE':       'mice',
    }

    norm_methods = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile', 'reference']
    n_replications = 20
    all_results = []

    for sev_label, lod_pct in lod_severities.items():
        for imp_label, imp_method in imputation_methods.items():
            print(f"  LOD={sev_label}, imputation={imp_label} ({n_replications} reps)...")
            for rep in range(n_replications):
                np.random.seed(42 + rep)
                params = {
                    'n_subjects': 100,
                    'n_biomarkers': 10,
                    'n_groups': 2,
                    'correlation_type': 'moderate',
                    'effect_size': 0.8,
                    'distribution': 'lognormal',
                    'dilution_alpha': 5.0,   # moderate dilution
                    'dilution_beta': 5.0,
                    'lod_percentile': lod_pct,
                    'lod_handling': imp_method,
                }
                try:
                    results = run_single_simulation(params, norm_methods)
                    for method in norm_methods:
                        row = {
                            'lod_severity': sev_label,
                            'imputation': imp_label,
                            'norm_method': method,
                            'replication': rep,
                            'power': results['univariate'][method]['power'],
                            'type_i_error': results['univariate'][method]['type_i_error'],
                            'corr_rank': results['correlation'][method]['rank_correlation'],
                            'pca_rv': results['pca'][method]['rv_coefficient'],
                            'clustering_ari': results['clustering'][method]['adjusted_rand_index'],
                            'classification_acc': results['classification'][method]['accuracy'],
                        }
                        all_results.append(row)
                except Exception as e:
                    print(f"    Warning: rep {rep} failed ({sev_label}, {imp_label}): {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'lod_sensitivity_results.csv', index=False)

    # ---- Figure E6: LOD sensitivity heatmaps ----
    # Aggregate across replications
    agg = df.groupby(['lod_severity', 'imputation', 'norm_method']).agg(
        power=('power', 'mean'),
        type_i_error=('type_i_error', 'mean'),
        corr_rank=('corr_rank', 'mean'),
        pca_rv=('pca_rv', 'mean'),
        clustering_ari=('clustering_ari', 'mean'),
        classification_acc=('classification_acc', 'mean'),
    ).reset_index()

    # Panel layout: 2 × 3 grid of heatmaps, each showing imputation × LOD severity
    # aggregated across normalization methods (mean)
    metrics_info = {
        'power': ('A) Statistical Power', 'YlGnBu'),
        'type_i_error': ('B) Type I Error', 'YlOrRd'),
        'corr_rank': ('C) Correlation Recovery', 'YlGnBu'),
        'pca_rv': ('D) PCA Structure (RV)', 'YlGnBu'),
        'clustering_ari': ('E) Clustering ARI', 'YlGnBu'),
        'classification_acc': ('F) Classification Accuracy', 'YlGnBu'),
    }

    sev_order = ['5%', '10%', '20%', '30%', '50%']
    imp_order = ['Zero', 'LOD/2', 'LOD/√2', 'LOD', 'MICE']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (metric, (title, cmap)) in enumerate(metrics_info.items()):
        ax = axes[idx // 3, idx % 3]
        # Average across normalization methods for overview
        pivot = agg.groupby(['lod_severity', 'imputation'])[metric].mean().reset_index()
        pivot_tbl = pivot.pivot(index='imputation', columns='lod_severity', values=metric)
        pivot_tbl = pivot_tbl.reindex(index=imp_order, columns=sev_order)
        sns.heatmap(pivot_tbl, ax=ax, annot=True, fmt='.3f', cmap=cmap,
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('LOD Severity (% below LOD)')
        ax.set_ylabel('Imputation Method')

    fig.suptitle('Figure E6: LOD Sensitivity — Imputation Method × Severity',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE6_lod_sensitivity.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE6_lod_sensitivity.png')
    plt.close(fig)

    # ---- Figure E7: LOD × normalization interaction (selected metric: power) ----
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
    for i, sev in enumerate(sev_order):
        ax = axes[i]
        sub = agg[agg['lod_severity'] == sev]
        pivot_tbl = sub.pivot(index='norm_method', columns='imputation', values='power')
        pivot_tbl = pivot_tbl.reindex(columns=imp_order)
        sns.heatmap(pivot_tbl, ax=ax, annot=True, fmt='.3f', cmap='YlGnBu',
                    linewidths=0.5, vmin=0, vmax=1, cbar=(i == 4))
        ax.set_title(f'LOD = {sev}', fontweight='bold')
        ax.set_xlabel('Imputation')
        if i == 0:
            ax.set_ylabel('Normalization Method')
        else:
            ax.set_ylabel('')

    fig.suptitle('Figure E7: Power by Normalization × Imputation at Each LOD Severity',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE7_lod_norm_interaction.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE7_lod_norm_interaction.png')
    plt.close(fig)

    # ---- Summary table ----
    summary = agg.groupby(['lod_severity', 'imputation']).agg(
        power=('power', 'mean'),
        type_i=('type_i_error', 'mean'),
        corr=('corr_rank', 'mean'),
        pca=('pca_rv', 'mean'),
        ari=('clustering_ari', 'mean'),
        acc=('classification_acc', 'mean'),
    ).round(3).reset_index()
    summary.to_csv(RESULTS_DIR / 'lod_sensitivity_summary.csv', index=False)

    print(f"  Saved: figE6_lod_sensitivity.pdf, figE7_lod_norm_interaction.pdf")
    print(f"  Saved: lod_sensitivity_results.csv, lod_sensitivity_summary.csv")
    print(f"  Total rows: {len(df)}")
    return df


def analysis_14_distance_metrics():
    """Evaluate distance matrix recovery under dilution (Figure 9, Figure E8)."""
    print("\n" + "=" * 60)
    print("Analysis 14: Distance Metric Recovery Under Dilution")
    print("=" * 60)

    np.random.seed(42)

    dilution_configs = {
        'Mild': {'dilution_alpha': 8.0, 'dilution_beta': 2.0},
        'Moderate': {'dilution_alpha': 5.0, 'dilution_beta': 5.0},
        'Severe': {'dilution_alpha': 2.0, 'dilution_beta': 8.0},
    }

    norm_methods = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile', 'reference']
    distance_metrics = ['euclidean', 'bray_curtis', 'aitchison', 'cosine',
                        'manhattan', 'canberra', 'mahalanobis']
    n_replications = 20
    all_results = []

    # Also store one example dataset for scatter plot
    example_scatter = None

    for severity, dil_params in dilution_configs.items():
        print(f"  Running {severity} dilution ({n_replications} replications)...")
        for rep in range(n_replications):
            np.random.seed(42 + rep)
            params = {
                'n_subjects': 100,
                'n_biomarkers': 10,
                'n_groups': 2,
                'correlation_type': 'moderate',
                'effect_size': 0.8,
                'distribution': 'lognormal',
                'lod_percentile': 0.1,
                'lod_handling': 'substitute',
                **dil_params,
            }
            try:
                dataset = generate_dataset(**params)
                X_true = dataset['X_true']
                X_obs = dataset['X_obs']
                y = dataset['y']

                for dist_metric in distance_metrics:
                    # True distance matrix (always computed on raw true data
                    # for the given metric)
                    try:
                        D_true = compute_distance_matrix(X_true, dist_metric)
                    except Exception:
                        continue

                    for norm_method in norm_methods:
                        # Bray-Curtis requires non-negative data; skip if CLR
                        if dist_metric == 'bray_curtis' and norm_method == 'clr':
                            continue
                        # Aitchison already applies CLR internally; skip CLR norm
                        if dist_metric == 'aitchison' and norm_method == 'clr':
                            continue
                        # Canberra requires non-negative data; skip if CLR
                        if dist_metric == 'canberra' and norm_method == 'clr':
                            continue

                        X_norm = normalize_data(X_obs, norm_method)

                        try:
                            # Compute observed distance matrix
                            D_obs = compute_distance_matrix(X_norm, dist_metric)

                            # Evaluate distance matrix recovery (use fewer perms for speed)
                            eval_res = evaluate_distance_matrix(D_true, D_obs, n_permutations=199)

                            # PERMANOVA R²
                            perm_res = permanova_r2(D_obs, y, n_permutations=199)
                        except Exception:
                            continue

                        row = {
                            'severity': severity,
                            'replication': rep,
                            'distance_metric': dist_metric,
                            'norm_method': norm_method,
                            'mantel_r': eval_res['mantel_r'],
                            'mantel_p': eval_res['mantel_p'],
                            'rank_correlation': eval_res['rank_correlation'],
                            'mean_relative_error': eval_res['mean_relative_error'],
                            'permanova_r2': perm_res['r2'],
                            'permanova_f': perm_res['pseudo_f'],
                            'permanova_p': perm_res['p_value'],
                        }
                        all_results.append(row)

                        # Save example for scatter plot (first rep, moderate severity)
                        if (example_scatter is None and severity == 'Moderate'
                                and rep == 0 and norm_method == 'none'):
                            example_scatter = {}
                            n = D_true.shape[0]
                            triu_idx = np.triu_indices(n, k=1)
                            for dm in distance_metrics:
                                try:
                                    D_t = compute_distance_matrix(X_true, dm)
                                    D_o = compute_distance_matrix(X_obs, dm)
                                    example_scatter[dm] = {
                                        'true': D_t[triu_idx],
                                        'observed': D_o[triu_idx],
                                    }
                                except Exception:
                                    pass

            except Exception as e:
                print(f"    Warning: replication {rep} failed for {severity}: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'distance_metrics_results.csv', index=False)

    # ---- Figure 9: Distance Summary (1×3 compact) ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel A: Mantel r by distance metric for key normalizations
    ax = axes[0]
    key_norms = ['none', 'clr', 'pqn']
    sub = df[df['norm_method'].isin(key_norms)].copy()
    sns.barplot(data=sub, x='distance_metric', y='mantel_r', hue='severity', ax=ax,
                palette={'Mild': '#2196F3', 'Moderate': '#FF9800', 'Severe': '#F44336'},
                ci=95, capsize=0.05)
    ax.set_title('A) Mantel Correlation by Distance Metric', fontweight='bold')
    ax.set_xlabel('Distance Metric')
    ax.set_ylabel('Mantel r')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Dilution', fontsize=7)

    # Panel B: PERMANOVA R² heatmap (norm × distance, averaged over severities)
    ax = axes[1]
    pivot = df.groupby(['norm_method', 'distance_metric'])['permanova_r2'].mean().reset_index()
    pivot_tbl = pivot.pivot(index='norm_method', columns='distance_metric', values='permanova_r2')
    norm_order = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile', 'reference']
    dm_order = ['euclidean', 'bray_curtis', 'aitchison', 'cosine',
                'manhattan', 'canberra', 'mahalanobis']
    pivot_tbl = pivot_tbl.reindex(index=[n for n in norm_order if n in pivot_tbl.index],
                                   columns=[d for d in dm_order if d in pivot_tbl.columns])
    sns.heatmap(pivot_tbl, ax=ax, annot=True, fmt='.3f', cmap='YlGnBu',
                linewidths=0.5, vmin=0, vmax=0.5, cbar_kws={'shrink': 0.8},
                annot_kws={'fontsize': 7})
    ax.set_title('B) PERMANOVA R² (mean across severities)', fontweight='bold')
    ax.set_xlabel('Distance Metric')
    ax.set_ylabel('Normalization')

    # Panel C: True vs Observed pairwise distances scatter
    ax = axes[2]
    if example_scatter is not None:
        colors_dm = {
            'euclidean': '#0072B2', 'bray_curtis': '#D55E00', 'aitchison': '#009E73',
            'cosine': '#CC79A7', 'manhattan': '#E69F00', 'canberra': '#56B4E9',
            'mahalanobis': '#F0E442',
        }
        for dm in distance_metrics:
            if dm not in example_scatter:
                continue
            d = example_scatter[dm]
            ax.scatter(d['true'], d['observed'], alpha=0.3, s=8,
                       c=colors_dm.get(dm, '#999999'),
                       label=dm.replace('_', '-').title(), edgecolors='none')
        scatter_dms = [dm for dm in distance_metrics if dm in example_scatter]
        if scatter_dms:
            max_val = max(max(example_scatter[dm]['true'].max(),
                              example_scatter[dm]['observed'].max()) for dm in scatter_dms)
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1, label='y = x')
        ax.set_xlabel('True Pairwise Distance')
        ax.set_ylabel('Observed Pairwise Distance')
        ax.legend(fontsize=6, markerscale=2)
    ax.set_title('C) True vs. Observed Distances (Moderate)', fontweight='bold')

    fig.suptitle('Figure 9: Distance Matrix Recovery Under Dilution',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig9_distance_summary.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'fig9_distance_summary.png')
    plt.close(fig)

    # ---- Figure E8: Detailed distance metrics (2×N grid) ----
    n_dm = len(distance_metrics)
    fig, axes = plt.subplots(2, n_dm, figsize=(6 * n_dm, 10))
    sev_palette = {'Mild': '#2196F3', 'Moderate': '#FF9800', 'Severe': '#F44336'}

    for col_idx, dm in enumerate(distance_metrics):
        sub_dm = df[df['distance_metric'] == dm].copy()

        # Row 1: Mantel r boxplots
        ax = axes[0, col_idx]
        if len(sub_dm) > 0:
            sns.boxplot(data=sub_dm, x='norm_method', y='mantel_r', hue='severity',
                        ax=ax, palette=sev_palette, fliersize=2)
        ax.set_title(f'Mantel r — {dm.replace("_", "-").title()}',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Normalization')
        ax.set_ylabel('Mantel r')
        ax.tick_params(axis='x', rotation=45)
        if col_idx == 0:
            ax.legend(title='Dilution', fontsize=6)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        # Row 2: PERMANOVA R² boxplots
        ax = axes[1, col_idx]
        if len(sub_dm) > 0:
            sns.boxplot(data=sub_dm, x='norm_method', y='permanova_r2', hue='severity',
                        ax=ax, palette=sev_palette, fliersize=2)
        ax.set_title(f'PERMANOVA R² — {dm.replace("_", "-").title()}',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Normalization')
        ax.set_ylabel('PERMANOVA R²')
        ax.tick_params(axis='x', rotation=45)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.suptitle('Figure E8: Distance Metric Recovery — Detailed Results',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE8_distance_metrics.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / 'figE8_distance_metrics.png')
    plt.close(fig)

    # Summary table
    summary = df.groupby(['severity', 'distance_metric', 'norm_method']).agg(
        mantel_r=('mantel_r', 'mean'),
        rank_corr=('rank_correlation', 'mean'),
        permanova_r2=('permanova_r2', 'mean'),
    ).round(3).reset_index()
    summary.to_csv(RESULTS_DIR / 'distance_metrics_summary.csv', index=False)

    print(f"  Saved: fig9_distance_summary.pdf, figE8_distance_metrics.pdf")
    print(f"  Saved: distance_metrics_results.csv, distance_metrics_summary.csv")
    print(f"  Total rows: {len(df)}")
    return df


def analysis_15_distribution_log_comparison():
    """Distribution × log-transform sensitivity analysis (Figure E22).

    Tests whether CLR's advantage comes from compositional normalization or
    simply from log-transforming right-skewed data.  Under log-normal data the
    log transform is inherently beneficial; under normal data it should not
    help (and may hurt).
    """
    print("\n" + "=" * 60)
    print("Analysis 15: Distribution × Log-Transform Comparison")
    print("=" * 60)

    np.random.seed(42)

    distributions = ['normal', 'lognormal']
    dilution_configs = {
        'Mild':     {'dilution_alpha': 8.0, 'dilution_beta': 2.0},
        'Moderate': {'dilution_alpha': 5.0, 'dilution_beta': 5.0},
        'Severe':   {'dilution_alpha': 2.0, 'dilution_beta': 8.0},
    }
    norm_methods = ['none', 'total_sum', 'pqn', 'clr', 'median', 'quantile']
    n_replications = 20
    eps = 1e-5
    all_results = []

    for dist in distributions:
        for severity, dil_params in dilution_configs.items():
            print(f"  {dist} / {severity} ({n_replications} reps)...")
            for rep in range(n_replications):
                np.random.seed(42 + rep)
                try:
                    dataset = generate_dataset(
                        n_subjects=100, n_biomarkers=10, n_groups=2,
                        correlation_type='moderate', effect_size=0.8,
                        distribution=dist, lod_percentile=0.1,
                        lod_handling='substitute', **dil_params,
                    )
                    X_obs  = dataset['X_obs']
                    y      = dataset['y']

                    # Truth from known group means (scale-independent)
                    truly_diff = dataset['truly_differential']
                    p_true_known = np.where(truly_diff, 0.0, 1.0)

                    for method in norm_methods:
                        X_norm = normalize_data(X_obs, method)

                        if method == 'clr':
                            scales = [('log', X_norm, p_true_known)]
                        else:
                            X_log = np.log(np.maximum(X_norm, 0) + eps)
                            scales = [
                                ('raw', X_norm,  p_true_known),
                                ('log', X_log,   p_true_known),
                            ]

                        for scale_label, X_eval, p_truth in scales:
                            # --- Univariate ---
                            p_obs, _ = analyze_univariate(X_eval, y, 't_test')
                            uni = evaluate_univariate(p_truth, p_obs)

                            # --- PERMANOVA R² (Euclidean) ---
                            D_euc = compute_distance_matrix(X_eval, 'euclidean')
                            perm_euc = permanova_r2(D_euc, y, n_permutations=199)

                            # --- PERMANOVA R² (Cosine) ---
                            try:
                                D_cos = compute_distance_matrix(X_eval, 'cosine')
                                perm_cos = permanova_r2(D_cos, y, n_permutations=199)
                                cos_r2 = perm_cos['r2']
                            except Exception:
                                cos_r2 = np.nan

                            # --- Clustering ARI ---
                            labels = analyze_clustering(X_eval, n_clusters=2)
                            clust = evaluate_clustering(y, labels)

                            # --- Classification accuracy (5-fold CV) ---
                            from sklearn.model_selection import StratifiedKFold
                            skf = StratifiedKFold(n_splits=5, shuffle=True,
                                                  random_state=42 + rep)
                            fold_accs = []
                            for train_idx, test_idx in skf.split(X_eval, y):
                                proba = analyze_classification(
                                    X_eval[train_idx], y[train_idx],
                                    X_eval[test_idx], 'logistic')
                                cls = evaluate_classification(y[test_idx], proba)
                                fold_accs.append(cls['accuracy'])
                            cv_acc = float(np.mean(fold_accs))

                            all_results.append({
                                'distribution': dist,
                                'severity': severity,
                                'replication': rep,
                                'norm_method': method,
                                'scale': scale_label,
                                'power': uni['power'],
                                'type_i_error': uni['type_i_error'],
                                'permanova_r2_euclidean': perm_euc['r2'],
                                'permanova_r2_cosine': cos_r2,
                                'clustering_ari': clust['adjusted_rand_index'],
                                'classification_acc': cv_acc,
                            })

                except Exception as e:
                    print(f"    Warning: rep {rep} failed ({dist}/{severity}): {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'distribution_log_comparison.csv', index=False)
    print(f"  Total rows: {len(df)}")

    # ---- Figure E22: 2×4 grid ----
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    metric_cols  = ['power', 'type_i_error', 'permanova_r2_euclidean',
                    'permanova_r2_cosine']
    metric_labels = ['Power', 'Type I Error', 'PERMANOVA R² (Euclidean)',
                     'PERMANOVA R² (Cosine)']
    dist_labels   = {'normal': 'Normal', 'lognormal': 'Log-Normal'}

    # Use Moderate severity only for the figure
    sub = df[df['severity'] == 'Moderate'].copy()

    for row_idx, dist in enumerate(distributions):
        sub_dist = sub[sub['distribution'] == dist]

        for col_idx, (mcol, mlab) in enumerate(zip(metric_cols, metric_labels)):
            ax = axes[row_idx, col_idx]

            # Separate CLR (always 'log' scale) and non-CLR methods
            non_clr = sub_dist[sub_dist['norm_method'] != 'clr']
            clr_vals = sub_dist[sub_dist['norm_method'] == 'clr'][mcol]
            clr_mean = clr_vals.mean() if len(clr_vals) > 0 else np.nan

            # Grouped bar plot: raw (orange) vs log (blue) for non-CLR
            method_order = ['none', 'total_sum', 'pqn', 'median', 'quantile']
            scale_palette = {'raw': '#FF9800', 'log': '#2196F3'}

            non_clr_ordered = non_clr[non_clr['norm_method'].isin(method_order)]
            sns.barplot(
                data=non_clr_ordered, x='norm_method', y=mcol, hue='scale',
                order=method_order, hue_order=['raw', 'log'],
                palette=scale_palette, ax=ax, ci='sd', capsize=0.04,
                errwidth=1.2)

            # CLR reference line
            ax.axhline(clr_mean, color='#4CAF50', linestyle='--',
                        linewidth=2, label=f'CLR ({clr_mean:.3f})')

            ax.set_title(f'{dist_labels[dist]} — {mlab}', fontweight='bold',
                         fontsize=9)
            ax.set_xlabel('Normalization')
            ax.set_ylabel(mlab)
            ax.tick_params(axis='x', rotation=45)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc='best')
            else:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

    fig.suptitle(
        'Figure E22: Distribution × Log-Transform Sensitivity\n'
        '(Moderate Dilution, 20 Replications)',
        fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE22_distribution_log_comparison.pdf',
                format='pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'figE22_distribution_log_comparison.png',
                bbox_inches='tight')
    plt.close(fig)

    # Summary table
    summary = df.groupby(['distribution', 'severity', 'norm_method', 'scale']).agg(
        power=('power', 'mean'),
        power_sd=('power', 'std'),
        type_i_error=('type_i_error', 'mean'),
        permanova_r2_euclidean=('permanova_r2_euclidean', 'mean'),
        permanova_r2_cosine=('permanova_r2_cosine', 'mean'),
        clustering_ari=('clustering_ari', 'mean'),
        classification_acc=('classification_acc', 'mean'),
    ).round(4).reset_index()
    summary.to_csv(RESULTS_DIR / 'distribution_log_comparison_summary.csv',
                   index=False)

    print(f"  Saved: figE22_distribution_log_comparison.pdf")
    print(f"  Saved: distribution_log_comparison.csv, "
          f"distribution_log_comparison_summary.csv")
    return df


def _analysis_16_pool_init():
    """Limit BLAS threads in worker processes to avoid contention."""
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    # Force OpenBLAS to respect the new setting
    try:
        import ctypes
        libopenblas = ctypes.CDLL("libopenblas64_.dylib")
        libopenblas.openblas_set_num_threads(1)
    except (OSError, AttributeError):
        pass


def _analysis_16_worker(args):
    """Worker for one (rge, rac, rep) combination in analysis_16."""
    rge, rac, rep = args
    norm_methods = ['none', 'total_sum', 'pqn', 'clr', 'median',
                    'quantile', 'reference']
    from sklearn.model_selection import StratifiedKFold
    results = []
    try:
        np.random.seed(42 + rep)
        dataset = generate_dataset(
            n_subjects=100, n_biomarkers=500, n_groups=2,
            correlation_type='moderate', effect_size=0.8,
            distribution='lognormal', lod_percentile=0.1,
            lod_handling='substitute',
            dilution_alpha=5.0, dilution_beta=5.0,
            ref_group_effect=rge, ref_analyte_corr=rac,
        )
        X_obs = dataset['X_obs']
        y = dataset['y']
        truly_diff = dataset['truly_differential']
        p_true = np.where(truly_diff, 0.0, 1.0)

        for method in norm_methods:
            X_norm = normalize_data(X_obs, method)

            p_obs, _ = analyze_univariate(X_norm, y, 't_test')
            uni = evaluate_univariate(p_true, p_obs)

            D_euc = compute_distance_matrix(X_norm, 'euclidean')
            perm_euc = permanova_r2(D_euc, y, n_permutations=199)

            labels = analyze_clustering(X_norm, n_clusters=2)
            clust = evaluate_clustering(y, labels)

            skf = StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=42 + rep)
            fold_accs = []
            for train_idx, test_idx in skf.split(X_norm, y):
                proba = analyze_classification(
                    X_norm[train_idx], y[train_idx],
                    X_norm[test_idx], 'logistic')
                cls = evaluate_classification(y[test_idx], proba)
                fold_accs.append(cls['accuracy'])

            results.append({
                'ref_group_effect': rge,
                'ref_analyte_corr': rac,
                'replication': rep,
                'norm_method': method,
                'power': uni['power'],
                'type_i_error': uni['type_i_error'],
                'permanova_r2': perm_euc['r2'],
                'clustering_ari': clust['adjusted_rand_index'],
                'classification_acc': float(np.mean(fold_accs)),
            })
    except Exception as e:
        print(f"  Warning: rep {rep} failed (rge={rge}/rac={rac}): {e}")
    return results


def analysis_16_correlated_reference():
    """Correlated reference biomarker simulation (Figure 10, Figure E23).

    Quantifies how reference normalization (dividing by total protein)
    degrades when the reference biomarker is itself elevated in the disease
    group (ref_group_effect) and correlated with analytes (ref_analyte_corr).
    """
    print("\n" + "=" * 60)
    print("Analysis 16: Correlated Reference Biomarker")
    print("=" * 60)

    ref_group_effects = [0.0, 0.25, 0.5, 0.75, 1.0]
    ref_analyte_corrs = [0.0, 0.3, 0.5, 0.7, 0.9]
    norm_methods = ['none', 'total_sum', 'pqn', 'clr', 'median',
                    'quantile', 'reference']
    n_replications = 20

    # Build task list for parallel execution
    tasks = [(rge, rac, rep)
             for rge in ref_group_effects
             for rac in ref_analyte_corrs
             for rep in range(n_replications)]

    n_workers = min(mp.cpu_count(), 8)
    print(f"  Running {len(tasks)} tasks across {n_workers} workers "
          f"(p=500 biomarkers)...")

    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers, initializer=_analysis_16_pool_init) as pool:
        nested_results = pool.map(_analysis_16_worker, tasks, chunksize=5)

    all_results = [r for batch in nested_results for r in batch]

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'correlated_reference_results.csv', index=False)
    print(f"  Total rows: {len(df)}")

    # ------------------------------------------------------------------
    # Figure 10: 2×3 main manuscript figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    method_palette = dict(zip(norm_methods, CB_PALETTE))

    # Panel A: Power vs ref_group_effect (ref_analyte_corr=0.7)
    ax = axes[0, 0]
    sub = df[df['ref_analyte_corr'] == 0.7]
    for m in norm_methods:
        ms = sub[sub['norm_method'] == m]
        means = ms.groupby('ref_group_effect')['power'].mean()
        ax.plot(means.index, means.values, 'o-',
                color=method_palette[m], label=m, linewidth=1.5)
    ax.set_xlabel('Reference Group Effect ($\\delta_{ref}$)')
    ax.set_ylabel('Power')
    ax.set_title('(A) Power vs Reference Group Effect\n'
                 '($\\rho_{ref}$ = 0.7)', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.set_ylim(-0.05, 1.05)

    # Panel B: Power vs ref_analyte_corr (ref_group_effect=0.5)
    ax = axes[0, 1]
    sub = df[df['ref_group_effect'] == 0.5]
    for m in norm_methods:
        ms = sub[sub['norm_method'] == m]
        means = ms.groupby('ref_analyte_corr')['power'].mean()
        ax.plot(means.index, means.values, 'o-',
                color=method_palette[m], label=m, linewidth=1.5)
    ax.set_xlabel('Reference-Analyte Correlation ($\\rho_{ref}$)')
    ax.set_ylabel('Power')
    ax.set_title('(B) Power vs Reference-Analyte Correlation\n'
                 '($\\delta_{ref}$ = 0.5)', fontweight='bold', fontsize=10)
    ax.set_ylim(-0.05, 1.05)

    # Panel C: Heatmap of reference norm power across all rge × rac
    ax = axes[0, 2]
    ref_only = df[df['norm_method'] == 'reference']
    hm_data = ref_only.groupby(
        ['ref_group_effect', 'ref_analyte_corr'])['power'].mean().reset_index()
    hm_pivot = hm_data.pivot(index='ref_group_effect',
                             columns='ref_analyte_corr', values='power')
    sns.heatmap(hm_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Power'})
    ax.set_xlabel('Reference-Analyte Correlation ($\\rho_{ref}$)')
    ax.set_ylabel('Reference Group Effect ($\\delta_{ref}$)')
    ax.set_title('(C) Reference Normalization Power\n(heatmap)',
                 fontweight='bold', fontsize=10)

    # Panel D: PERMANOVA R² vs ref_group_effect (ref_analyte_corr=0.7)
    ax = axes[1, 0]
    sub = df[df['ref_analyte_corr'] == 0.7]
    for m in norm_methods:
        ms = sub[sub['norm_method'] == m]
        means = ms.groupby('ref_group_effect')['permanova_r2'].mean()
        ax.plot(means.index, means.values, 'o-',
                color=method_palette[m], label=m, linewidth=1.5)
    ax.set_xlabel('Reference Group Effect ($\\delta_{ref}$)')
    ax.set_ylabel('PERMANOVA $R^2$')
    ax.set_title('(D) PERMANOVA $R^2$ vs Reference Group Effect\n'
                 '($\\rho_{ref}$ = 0.7)', fontweight='bold', fontsize=10)

    # Panel E: Type I error vs ref_group_effect (ref_analyte_corr=0.7)
    ax = axes[1, 1]
    sub = df[df['ref_analyte_corr'] == 0.7]
    for m in norm_methods:
        ms = sub[sub['norm_method'] == m]
        means = ms.groupby('ref_group_effect')['type_i_error'].mean()
        ax.plot(means.index, means.values, 'o-',
                color=method_palette[m], label=m, linewidth=1.5)
    ax.axhline(0.05, color='gray', linestyle='--', linewidth=1,
               label='$\\alpha$ = 0.05')
    ax.set_xlabel('Reference Group Effect ($\\delta_{ref}$)')
    ax.set_ylabel('Type I Error')
    ax.set_title('(E) Type I Error vs Reference Group Effect\n'
                 '($\\rho_{ref}$ = 0.7)', fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='best')

    # Panel F: Bar chart at worst case (rge=1.0, rac=0.9)
    ax = axes[1, 2]
    worst = df[(df['ref_group_effect'] == 1.0) &
               (df['ref_analyte_corr'] == 0.9)]
    worst_means = worst.groupby('norm_method')['power'].mean()
    worst_sds = worst.groupby('norm_method')['power'].std()
    bar_order = norm_methods
    bar_colors = [method_palette[m] for m in bar_order]
    x_pos = np.arange(len(bar_order))
    ax.bar(x_pos, [worst_means.get(m, 0) for m in bar_order],
           yerr=[worst_sds.get(m, 0) for m in bar_order],
           color=bar_colors, capsize=3, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_order, rotation=45, ha='right')
    ax.set_ylabel('Power')
    ax.set_title('(F) All Methods at Worst Case\n'
                 '($\\delta_{ref}$ = 1.0, $\\rho_{ref}$ = 0.9)',
                 fontweight='bold', fontsize=10)
    ax.set_ylim(0, 1.05)

    fig.suptitle('Figure 10: Correlated Reference Biomarker Simulation',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig10_correlated_reference.pdf',
                format='pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'fig10_correlated_reference.png',
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure E23: 5×4 supplement grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(5, 4, figsize=(22, 25))
    metric_cols = ['power', 'type_i_error', 'permanova_r2',
                   'classification_acc']
    metric_labels = ['Power', 'Type I Error', 'PERMANOVA $R^2$',
                     'Classification Accuracy']

    for row_idx, rac_val in enumerate(ref_analyte_corrs):
        sub_rac = df[df['ref_analyte_corr'] == rac_val]
        for col_idx, (mcol, mlab) in enumerate(
                zip(metric_cols, metric_labels)):
            ax = axes[row_idx, col_idx]
            for m in norm_methods:
                ms = sub_rac[sub_rac['norm_method'] == m]
                means = ms.groupby('ref_group_effect')[mcol].mean()
                ax.plot(means.index, means.values, 'o-',
                        color=method_palette[m], label=m, linewidth=1.2)
            if mcol == 'type_i_error':
                ax.axhline(0.05, color='gray', linestyle='--',
                           linewidth=1)
            ax.set_xlabel('$\\delta_{ref}$')
            ax.set_ylabel(mlab)
            ax.set_title(f'$\\rho_{{ref}}$ = {rac_val}',
                         fontweight='bold', fontsize=9)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=6, loc='best')

    fig.suptitle(
        'Figure E23: Correlated Reference — Detailed Results\n'
        '(Rows: $\\rho_{ref}$ = 0.0, 0.3, 0.5, 0.7, 0.9; '
        'Cols: power, Type I error, PERMANOVA $R^2$, classification)',
        fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'figE23_correlated_reference_detail.pdf',
                format='pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'figE23_correlated_reference_detail.png',
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    summary = df.groupby(
        ['ref_group_effect', 'ref_analyte_corr', 'norm_method']).agg(
        power=('power', 'mean'),
        power_sd=('power', 'std'),
        type_i_error=('type_i_error', 'mean'),
        type_i_error_sd=('type_i_error', 'std'),
        permanova_r2=('permanova_r2', 'mean'),
        clustering_ari=('clustering_ari', 'mean'),
        classification_acc=('classification_acc', 'mean'),
    ).round(4).reset_index()
    summary.to_csv(RESULTS_DIR / 'correlated_reference_summary.csv',
                   index=False)

    print(f"  Saved: fig10_correlated_reference.pdf")
    print(f"  Saved: figE23_correlated_reference_detail.pdf")
    print(f"  Saved: correlated_reference_results.csv, "
          f"correlated_reference_summary.csv")
    return df


def main():
    """Run all analyses."""
    print("=" * 60)
    print("BIOMARKER DILUTION SIMULATION STUDY")
    print("Complete Analysis Pipeline")
    print("=" * 60)

    start = time.time()

    # Run all analyses (original 8)
    dilution_models = analysis_1_dilution_models()
    dataset = analysis_2_dilution_impact_on_data()
    norm_df = analysis_3_normalization_comparison()
    sample_df = analysis_4_sample_size_effect()
    ml_df = analysis_5_ml_robustness()
    fs_df = analysis_6_feature_selection()
    power_df = analysis_7_power_analysis()
    stats_summary = analysis_8_enhanced_statistical()

    # New extended analyses (9-13)
    highdim_df = analysis_9_high_dimensional()
    type1_df = analysis_10_type1_error_decomposition()
    multigroup_df = analysis_11_multigroup()
    batch_df = analysis_12_batch_dilution_interaction()
    lod_df = analysis_13_lod_sensitivity()

    # Distance metrics analysis (14)
    dist_df = analysis_14_distance_metrics()

    # Distribution × log-transform comparison (15)
    distlog_df = analysis_15_distribution_log_comparison()

    # Correlated reference biomarker (16)
    corref_df = analysis_16_correlated_reference()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"ALL ANALYSES COMPLETE ({elapsed:.1f}s)")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 60)

    # Sync figures to manuscript directory
    manuscript_fig_dir = Path('manuscript_latex/figures')
    if manuscript_fig_dir.exists():
        for f in FIGURES_DIR.glob('*'):
            shutil.copy2(f, manuscript_fig_dir / f.name)
        print(f"Figures synced to {manuscript_fig_dir}/")

    # Summary of all outputs
    print("\nGenerated Figures:")
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f"  {f.name}")

    print("\nGenerated Data Tables:")
    for f in sorted(RESULTS_DIR.glob('*')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
