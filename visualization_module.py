import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple

# Import simulation functions (assuming they're in a module named biomarker_dilution_sim)
# For a standalone script, these would be directly imported
from biomarker_dilution_sim import *


def visualize_dilution_impact(
    n_subjects=200,
    n_biomarkers=5,
    n_groups=2,
    correlation_type='moderate',
    dilution_alpha=2.0,
    dilution_beta=5.0
):
    """
    Create visualizations showing the impact of dilution on various statistical properties.
    """
    # Generate dataset
    params = {
        'n_subjects': n_subjects,
        'n_biomarkers': n_biomarkers,
        'n_groups': n_groups,
        'correlation_type': correlation_type,
        'effect_size': 0.8,
        'distribution': 'lognormal',
        'dilution_alpha': dilution_alpha,
        'dilution_beta': dilution_beta,
        'lod_percentile': 0.1
    }
    
    dataset = generate_dataset(**params)
    
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    y = dataset['y']
    dilution_factors = dataset['dilution_factors']
    
    # Create a figure with multiple panels
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # 1. Scatter plot of true vs observed values
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(X_true[:, 0], X_obs[:, 0], c=dilution_factors, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('True Concentration')
    ax1.set_ylabel('Observed Concentration')
    ax1.set_title('True vs Observed Concentration')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Dilution Factor')
    
    # Add line y = x for reference
    max_val = max(X_true[:, 0].max(), X_obs[:, 0].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    # 2. Boxplots by group before and after dilution
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepare data for boxplot
    group_data_true = [X_true[y == i, 0] for i in range(n_groups)]
    group_data_obs = [X_obs[y == i, 0] for i in range(n_groups)]
    
    # Plot boxplots
    positions = np.arange(n_groups) * 2
    box1 = ax2.boxplot(group_data_true, positions=positions, patch_artist=True,
                      boxprops=dict(facecolor='lightblue'))
    box2 = ax2.boxplot(group_data_obs, positions=positions + 1, patch_artist=True,
                      boxprops=dict(facecolor='salmon'))
    
    # Add legend
    ax2.legend([box1["boxes"][0], box2["boxes"][0]], ['True', 'Observed'], loc='upper right')
    
    # Set labels
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Concentration')
    ax2.set_title('Group Differences Before and After Dilution')
    ax2.set_xticks(positions + 0.5)
    ax2.set_xticklabels([f'Group {i}' for i in range(n_groups)])
    
    # 3. PCA visualization before and after dilution
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Apply PCA to true and observed data
    pca_true = PCA(n_components=2)
    pca_obs = PCA(n_components=2)
    
    X_pca_true = pca_true.fit_transform(X_true)
    X_pca_obs = pca_obs.fit_transform(X_obs)
    
    # Plot PCA results
    for i in range(n_groups):
        # True data
        ax3.scatter(X_pca_true[y == i, 0], X_pca_true[y == i, 1], 
                   marker='o', alpha=0.7, label=f'True Group {i}')
        
        # Observed data
        ax3.scatter(X_pca_obs[y == i, 0], X_pca_obs[y == i, 1], 
                   marker='x', alpha=0.7, label=f'Obs Group {i}')
    
    # Add labels
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('PCA: True vs Observed Data')
    ax3.legend()
    
    # 4. Correlation heatmaps before and after dilution
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Calculate correlation matrices
    corr_true = np.corrcoef(X_true.T)
    corr_obs = np.corrcoef(X_obs.T)
    
    # Plot heatmaps
    sns.heatmap(corr_true, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax4)
    ax4.set_title('Correlation Matrix - True Data')
    
    sns.heatmap(corr_obs, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax5)
    ax5.set_title('Correlation Matrix - Observed Data')
    
    # 5. Impact of dilution on statistical tests
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Run t-tests on each biomarker for true and observed data
    p_values_true = []
    p_values_obs = []
    
    for j in range(n_biomarkers):
        # True data
        t_stat, p_val = stats.ttest_ind(
            X_true[y == 0, j], 
            X_true[y == 1, j],
            equal_var=False
        )
        p_values_true.append(p_val)
        
        # Observed data
        t_stat, p_val = stats.ttest_ind(
            X_obs[y == 0, j], 
            X_obs[y == 1, j],
            equal_var=False
        )
        p_values_obs.append(p_val)
    
    # Create bar plot of -log10(p) values
    biomarker_idx = np.arange(n_biomarkers)
    width = 0.35
    
    ax6.bar(biomarker_idx - width/2, -np.log10(p_values_true), width, label='True')
    ax6.bar(biomarker_idx + width/2, -np.log10(p_values_obs), width, label='Observed')
    
    # Add significance threshold line (p = 0.05)
    ax6.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.7)
    
    # Add labels
    ax6.set_xlabel('Biomarker')
    ax6.set_ylabel('-log10(p-value)')
    ax6.set_title('Statistical Significance Before and After Dilution')
    ax6.set_xticks(biomarker_idx)
    ax6.set_xticklabels([f'B{i+1}' for i in range(n_biomarkers)])
    ax6.legend()
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def visualize_normalization_methods(
    n_subjects=200,
    n_biomarkers=5,
    n_groups=2,
    dilution_alpha=2.0,
    dilution_beta=5.0
):
    """
    Visualize the effect of different normalization methods on diluted data.
    """
    # Generate dataset
    params = {
        'n_subjects': n_subjects,
        'n_biomarkers': n_biomarkers,
        'n_groups': n_groups,
        'correlation_type': 'moderate',
        'effect_size': 0.8,
        'distribution': 'lognormal',
        'dilution_alpha': dilution_alpha,
        'dilution_beta': dilution_beta,
        'lod_percentile': 0.1
    }
    
    dataset = generate_dataset(**params)
    
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    y = dataset['y']
    dilution_factors = dataset['dilution_factors']
    
    # Apply normalization methods
    methods = {
        'None': X_obs,
        'Total Sum': normalize_total_sum(X_obs),
        'PQN': normalize_probabilistic_quotient(X_obs),
        'CLR': centered_log_ratio(X_obs),
        'Reference': normalize_reference_biomarker(X_obs, 0)
    }
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(len(methods), 3, figure=fig)
    
    # For each normalization method
    for i, (method_name, X_norm) in enumerate(methods.items()):
        # 1. Scatter plot of first two biomarkers
        ax1 = fig.add_subplot(gs[i, 0])
        scatter = ax1.scatter(X_norm[:, 0], X_norm[:, 1], c=dilution_factors, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Biomarker 1')
        ax1.set_ylabel('Biomarker 2')
        ax1.set_title(f'{method_name}: Biomarker 1 vs 2')
        
        if i == 0:
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Dilution Factor')
        
        # 2. PCA visualization
        ax2 = fig.add_subplot(gs[i, 1])
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_norm)
        
        # Plot PCA results, colored by group
        for group in range(n_groups):
            ax2.scatter(X_pca[y == group, 0], X_pca[y == group, 1], 
                       alpha=0.7, label=f'Group {group}')
        
        # Add variation explained info
        var_explained = pca.explained_variance_ratio_
        ax2.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
        ax2.set_title(f'{method_name}: PCA')
        
        if i == 0:
            ax2.legend()
        
        # 3. Statistical tests
        ax3 = fig.add_subplot(gs[i, 2])
        
        # Compute p-values for true and normalized data
        p_values_true = []
        p_values_norm = []
        
        for j in range(n_biomarkers):
            # True data
            t_stat, p_val = stats.ttest_ind(
                X_true[y == 0, j], 
                X_true[y == 1, j],
                equal_var=False
            )
            p_values_true.append(p_val)
            
            # Normalized data
            if method_name == 'CLR' and j >= X_norm.shape[1]:
                # CLR reduces dimensionality in some implementations
                p_values_norm.append(1.0)
            else:
                t_stat, p_val = stats.ttest_ind(
                    X_norm[y == 0, j], 
                    X_norm[y == 1, j],
                    equal_var=False
                )
                p_values_norm.append(p_val)
        
        # Create bar plot of -log10(p) values
        biomarker_idx = np.arange(n_biomarkers)
        width = 0.35
        
        ax3.bar(biomarker_idx - width/2, -np.log10(p_values_true), width, label='True')
        ax3.bar(biomarker_idx + width/2, -np.log10(p_values_norm), width, label='Normalized')
        
        # Add significance threshold line (p = 0.05)
        ax3.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.7)
        
        # Add labels
        ax3.set_xlabel('Biomarker')
        ax3.set_ylabel('-log10(p-value)')
        ax3.set_title(f'{method_name}: Statistical Tests')
        ax3.set_xticks(biomarker_idx)
        ax3.set_xticklabels([f'B{i+1}' for i in range(n_biomarkers)])
        
        if i == 0:
            ax3.legend()
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def visualize_simulation_results(results_df):
    """
    Create visualizations summarizing Monte Carlo simulation results.
    """
    # Create figure
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # 1. Effect of dilution severity on different metrics
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create dilution severity category
    results_df['dilution_severity'] = np.where(
        (results_df['dilution_alpha'] > results_df['dilution_beta']),
        'Mild', 
        np.where(
            (results_df['dilution_alpha'] == results_df['dilution_beta']),
            'Moderate',
            'Severe'
        )
    )
    
    # Boxplot of power by dilution severity and normalization
    sns.boxplot(
        x='dilution_severity',
        y='power',
        hue='normalization',
        data=results_df,
        ax=ax1
    )
    ax1.set_title('Statistical Power by Dilution Severity and Normalization')
    ax1.set_xlabel('Dilution Severity')
    ax1.set_ylabel('Power')
    ax1.legend(title='Normalization')
    
    # 2. Effect of number of biomarkers on different metrics
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Line plot of average power by number of biomarkers and normalization
    sns.lineplot(
        x='n_biomarkers',
        y='power',
        hue='normalization',
        data=results_df,
        ax=ax2,
        markers=True,
        err_style='band'
    )
    ax2.set_title('Statistical Power by Number of Biomarkers and Normalization')
    ax2.set_xlabel('Number of Biomarkers')
    ax2.set_ylabel('Power')
    ax2.legend(title='Normalization')
    
    # 3. Correlation recovery performance
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Boxplot of correlation recovery by normalization
    sns.boxplot(
        x='normalization',
        y='corr_rank_correlation',
        data=results_df,
        ax=ax3
    )
    ax3.set_title('Correlation Recovery by Normalization Method')
    ax3.set_xlabel('Normalization Method')
    ax3.set_ylabel('Rank Correlation with True Correlations')
    
    # 4. PCA performance
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Boxplot of PCA recovery by normalization
    sns.boxplot(
        x='normalization',
        y='pca_rv_coefficient',
        data=results_df,
        ax=ax4
    )
    ax4.set_title('PCA Subspace Recovery by Normalization Method')
    ax4.set_xlabel('Normalization Method')
    ax4.set_ylabel('RV Coefficient')
    
    # 5. Clustering performance
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Boxplot of clustering performance by normalization
    sns.boxplot(
        x='normalization',
        y='clustering_ari',
        data=results_df,
        ax=ax5
    )
    ax5.set_title('Clustering Performance by Normalization Method')
    ax5.set_xlabel('Normalization Method')
    ax5.set_ylabel('Adjusted Rand Index')
    
    # 6. Classification performance
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Boxplot of classification performance by normalization
    sns.boxplot(
        x='normalization',
        y='classification_accuracy',
        data=results_df,
        ax=ax6
    )
    ax6.set_title('Classification Performance by Normalization Method')
    ax6.set_xlabel('Normalization Method')
    ax6.set_ylabel('Accuracy')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def visualize_dilution_recovery(dataset_params=None):
    """
    Visualize how well different normalization methods recover the true structure
    despite dilution effects.
    """
    if dataset_params is None:
        # Default parameters
        dataset_params = {
            'n_subjects': 200,
            'n_biomarkers': 10,
            'n_groups': 3,
            'correlation_type': 'block',
            'block_size': 3,
            'effect_size': 0.8,
            'distribution': 'lognormal',
            'dilution_alpha': 2.0,
            'dilution_beta': 8.0,
            'lod_percentile': 0.1
        }
    
    # Generate dataset
    dataset = generate_dataset(**dataset_params)
    
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    y = dataset['y']
    dilution_factors = dataset['dilution_factors']
    
    # Apply normalization methods
    norm_methods = {
        'True Data': X_true,
        'Diluted (Raw)': X_obs,
        'Total Sum': normalize_total_sum(X_obs),
        'PQN': normalize_probabilistic_quotient(X_obs),
        'CLR': centered_log_ratio(X_obs)
    }
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, len(norm_methods), figure=fig)
    
    # Create a custom colormap for groups
    cmap = plt.cm.get_cmap('tab10', len(np.unique(y)))
    
    # For each method
    for i, (method_name, X_data) in enumerate(norm_methods.items()):
        # 1. PCA visualization
        ax1 = fig.add_subplot(gs[0, i])
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_data)
        
        # Plot PCA results
        for g in np.unique(y):
            idx = y == g
            ax1.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Group {g}',
                       color=cmap(g), alpha=0.7)
        
        # Add labels
        var_explained = pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
        ax1.set_title(f'{method_name} - PCA')
        
        if i == 0:
            ax1.legend()
        
        # 2. Correlation heatmap
        ax2 = fig.add_subplot(gs[1, i])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_data.T)
        
        # Plot heatmap
        sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, ax=ax2)
        ax2.set_title(f'{method_name} - Correlation')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Visualize dilution impact
    fig1 = visualize_dilution_impact()
    fig1.savefig('dilution_impact.png')
    
    # Visualize normalization methods
    fig2 = visualize_normalization_methods()
    fig2.savefig('normalization_methods.png')
    
    # Visualize dilution recovery
    fig3 = visualize_dilution_recovery()
    fig3.savefig('dilution_recovery.png')
    
    plt.show()
