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

warnings.filterwarnings('ignore')

from biomarker_dilution_sim import (
    generate_dataset, generate_dilution_factors,
    generate_dilution_factors_time_dependent,
    generate_dilution_factors_covariate_dependent,
    normalize_data,
    analyze_univariate, analyze_univariate_enhanced,
    analyze_correlation, analyze_pca, analyze_clustering,
    analyze_classification, analyze_classification_advanced,
    feature_selection, cross_validate_classification,
    multiple_testing_correction, calculate_effect_size,
    bootstrap_confidence_interval, power_analysis, sample_size_estimation,
    evaluate_univariate, evaluate_correlation, evaluate_pca,
    evaluate_clustering, evaluate_classification,
    run_single_simulation
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

# Plotting defaults
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})


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
        sns.boxplot(data=subset, x='severity', y='jaccard', hue='normalization',
                    ax=ax, palette='Set2', order=['Mild', 'Moderate', 'Severe'],
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


def main():
    """Run all analyses."""
    print("=" * 60)
    print("BIOMARKER DILUTION SIMULATION STUDY")
    print("Complete Analysis Pipeline")
    print("=" * 60)

    start = time.time()

    # Run all analyses
    dilution_models = analysis_1_dilution_models()
    dataset = analysis_2_dilution_impact_on_data()
    norm_df = analysis_3_normalization_comparison()
    sample_df = analysis_4_sample_size_effect()
    ml_df = analysis_5_ml_robustness()
    fs_df = analysis_6_feature_selection()
    power_df = analysis_7_power_analysis()
    stats_summary = analysis_8_enhanced_statistical()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"ALL ANALYSES COMPLETE ({elapsed:.1f}s)")
    print(f"Figures saved to: {FIGURES_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("=" * 60)

    # Summary of all outputs
    print("\nGenerated Figures:")
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f"  {f.name}")

    print("\nGenerated Data Tables:")
    for f in sorted(RESULTS_DIR.glob('*')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
