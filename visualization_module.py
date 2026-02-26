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


def plot_volcano(
    fold_changes: np.ndarray,
    p_values: np.ndarray,
    biomarker_names: List[str] = None,
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
    title: str = "Volcano Plot",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create a volcano plot for differential expression analysis.

    Parameters:
    -----------
    fold_changes : np.ndarray
        Log2 fold changes
    p_values : np.ndarray
        P-values (will be -log10 transformed)
    biomarker_names : list, optional
        Names of biomarkers for labeling
    fc_threshold : float
        Fold change threshold for significance
    p_threshold : float
        P-value threshold for significance
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Transform p-values
    neg_log_p = -np.log10(p_values + 1e-300)

    # Determine significance
    significant_up = (fold_changes > fc_threshold) & (p_values < p_threshold)
    significant_down = (fold_changes < -fc_threshold) & (p_values < p_threshold)
    not_significant = ~(significant_up | significant_down)

    # Plot points
    ax.scatter(fold_changes[not_significant], neg_log_p[not_significant],
               c='gray', alpha=0.5, s=50, label='Not significant')
    ax.scatter(fold_changes[significant_up], neg_log_p[significant_up],
               c='red', alpha=0.7, s=50, label='Upregulated')
    ax.scatter(fold_changes[significant_down], neg_log_p[significant_down],
               c='blue', alpha=0.7, s=50, label='Downregulated')

    # Add threshold lines
    ax.axhline(y=-np.log10(p_threshold), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=fc_threshold, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-fc_threshold, color='gray', linestyle='--', alpha=0.5)

    # Label significant points
    if biomarker_names is not None:
        significant = significant_up | significant_down
        for i in np.where(significant)[0]:
            ax.annotate(biomarker_names[i],
                       (fold_changes[i], neg_log_p[i]),
                       fontsize=8, alpha=0.8)

    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10 P-value')
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_roc_curves_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    title: str = "ROC Curve Comparison",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple methods/normalization approaches.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    predictions_dict : dict
        Dictionary of {method_name: predicted_probabilities}
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(predictions_dict)))

    for (method_name, y_proba), color in zip(predictions_dict.items(), colors):
        # Handle multi-class by using one-vs-rest for first class
        if len(y_proba.shape) > 1:
            y_score = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, 0]
        else:
            y_score = y_proba

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{method_name} (AUC = {roc_auc:.3f})')

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')

    fig.tight_layout()
    return fig


def plot_pca_3d(
    X: np.ndarray,
    y: np.ndarray,
    dilution_factors: np.ndarray = None,
    title: str = "3D PCA Visualization",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create 3D PCA visualization.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    y : np.ndarray
        Group labels
    dilution_factors : np.ndarray, optional
        Dilution factors for coloring
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Perform PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    unique_groups = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

    for group, color in zip(unique_groups, colors):
        mask = y == group
        if dilution_factors is not None:
            scatter = ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                c=dilution_factors[mask], cmap='viridis',
                s=50, alpha=0.7, label=f'Group {group}'
            )
        else:
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                c=[color], s=50, alpha=0.7, label=f'Group {group}'
            )

    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
    ax.set_zlabel(f'PC3 ({var_explained[2]:.1%})')
    ax.set_title(title)
    ax.legend()

    if dilution_factors is not None:
        fig.colorbar(scatter, ax=ax, label='Dilution Factor', shrink=0.5)

    return fig


def plot_tsne_2d(
    X: np.ndarray,
    y: np.ndarray,
    perplexity: int = 30,
    title: str = "t-SNE Visualization",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create t-SNE visualization.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    y : np.ndarray
        Group labels
    perplexity : int
        t-SNE perplexity parameter
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)

    unique_groups = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

    for group, color in zip(unique_groups, colors):
        mask = y == group
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                  c=[color], s=50, alpha=0.7, label=f'Group {group}')

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_forest(
    effect_sizes: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    biomarker_names: List[str] = None,
    title: str = "Forest Plot - Effect Sizes",
    figsize: Tuple[int, int] = (10, 12)
) -> plt.Figure:
    """
    Create a forest plot for effect sizes with confidence intervals.

    Parameters:
    -----------
    effect_sizes : np.ndarray
        Point estimates of effect sizes
    ci_lower : np.ndarray
        Lower confidence interval bounds
    ci_upper : np.ndarray
        Upper confidence interval bounds
    biomarker_names : list, optional
        Names of biomarkers
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    n_biomarkers = len(effect_sizes)

    if biomarker_names is None:
        biomarker_names = [f'Biomarker {i+1}' for i in range(n_biomarkers)]

    # Sort by effect size
    sorted_idx = np.argsort(effect_sizes)

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(n_biomarkers)

    # Plot confidence intervals
    for i, idx in enumerate(sorted_idx):
        color = 'red' if effect_sizes[idx] > 0 else 'blue'
        ax.plot([ci_lower[idx], ci_upper[idx]], [i, i],
                color=color, linewidth=2, alpha=0.7)
        ax.scatter(effect_sizes[idx], i, color=color, s=80, zorder=5)

    # Reference line at zero
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([biomarker_names[i] for i in sorted_idx])
    ax.set_xlabel('Effect Size (Cohen\'s d)')
    ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_heatmap_clustered(
    X: np.ndarray,
    y: np.ndarray = None,
    biomarker_names: List[str] = None,
    sample_names: List[str] = None,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    title: str = "Clustered Heatmap",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create a clustered heatmap of biomarker data.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (samples x biomarkers)
    y : np.ndarray, optional
        Group labels for annotation
    biomarker_names : list, optional
        Names of biomarkers
    sample_names : list, optional
        Names of samples
    cluster_rows : bool
        Whether to cluster rows
    cluster_cols : bool
        Whether to cluster columns
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    n_samples, n_biomarkers = X.shape

    if biomarker_names is None:
        biomarker_names = [f'B{i+1}' for i in range(n_biomarkers)]
    if sample_names is None:
        sample_names = [f'S{i+1}' for i in range(n_samples)]

    # Standardize data for visualization
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)

    # Clustering
    if cluster_rows:
        row_linkage = linkage(pdist(X_std), method='average')
        row_order = dendrogram(row_linkage, no_plot=True)['leaves']
    else:
        row_order = list(range(n_samples))

    if cluster_cols:
        col_linkage = linkage(pdist(X_std.T), method='average')
        col_order = dendrogram(col_linkage, no_plot=True)['leaves']
    else:
        col_order = list(range(n_biomarkers))

    # Reorder data
    X_ordered = X_std[row_order, :][:, col_order]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(X_ordered, aspect='auto', cmap='RdBu_r',
                   vmin=-3, vmax=3)

    # Labels
    ax.set_xticks(np.arange(n_biomarkers))
    ax.set_xticklabels([biomarker_names[i] for i in col_order], rotation=45, ha='right')

    if n_samples <= 50:
        ax.set_yticks(np.arange(n_samples))
        ax.set_yticklabels([sample_names[i] for i in row_order])
    else:
        ax.set_yticks([])

    ax.set_title(title)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Z-score')

    # Add group annotation if provided
    if y is not None:
        # Create color bar for groups
        group_colors = plt.cm.Set1(y[row_order] / (max(y) + 1))
        for i, color in enumerate(group_colors):
            ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 0.8, 1,
                                       facecolor=color, edgecolor='none'))

    fig.tight_layout()
    return fig


def plot_method_comparison_radar(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    title: str = "Method Comparison",
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Create a radar/spider plot comparing methods across multiple metrics.

    Parameters:
    -----------
    metrics_dict : dict
        {method_name: {metric_name: value}}
    metric_names : list, optional
        List of metrics to include
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    methods = list(metrics_dict.keys())

    if metric_names is None:
        metric_names = list(metrics_dict[methods[0]].keys())

    n_metrics = len(metric_names)

    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for method, color in zip(methods, colors):
        values = [metrics_dict[method].get(m, 0) for m in metric_names]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=method)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    fig.tight_layout()
    return fig


def plot_dilution_distribution_comparison(
    dilution_data: Dict[str, np.ndarray],
    title: str = "Dilution Factor Distributions",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Compare dilution factor distributions from different models.

    Parameters:
    -----------
    dilution_data : dict
        {distribution_name: dilution_factors_array}
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    n_distributions = len(dilution_data)

    fig, axes = plt.subplots(1, n_distributions, figsize=figsize, sharey=True)

    if n_distributions == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, n_distributions))

    for ax, (name, factors), color in zip(axes, dilution_data.items(), colors):
        ax.hist(factors, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(factors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(factors):.3f}')
        ax.axvline(np.median(factors), color='blue', linestyle=':',
                   label=f'Median: {np.median(factors):.3f}')
        ax.set_xlabel('Dilution Factor')
        ax.set_title(name)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Count')
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_power_curve(
    effect_sizes: np.ndarray,
    sample_sizes: np.ndarray,
    alpha: float = 0.05,
    title: str = "Power Analysis",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot power curves for different effect sizes and sample sizes.

    Parameters:
    -----------
    effect_sizes : np.ndarray
        Array of effect sizes to evaluate
    sample_sizes : np.ndarray
        Array of sample sizes per group
    alpha : float
        Significance level
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    from biomarker_dilution_sim import power_analysis

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 1, len(effect_sizes)))

    for es, color in zip(effect_sizes, colors):
        powers = [power_analysis(es, n, alpha) for n in sample_sizes]
        ax.plot(sample_sizes, powers, 'o-', color=color,
                label=f'd = {es:.2f}', linewidth=2)

    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% power')
    ax.set_xlabel('Sample Size per Group')
    ax.set_ylabel('Statistical Power')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

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
