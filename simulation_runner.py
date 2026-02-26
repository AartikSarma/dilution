import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime
import time
import argparse
from pathlib import Path

# Import simulation modules
from biomarker_dilution_sim import *
from visualization_module import *
from data_io import (
    SimulationConfig, ExperimentTracker, ReproducibilityManager,
    save_results, results_to_dataframe, create_experiment
)

def run_demo_simulation():
    """
    Run a small demonstration simulation with a limited parameter set.
    """
    print("Running demonstration simulation...")
    
    # Set random seed
    np.random.seed(42)
    
    # Set parameters
    params = {
        'n_subjects': 100,
        'n_biomarkers': 10,
        'n_groups': 2,
        'correlation_type': 'moderate',
        'effect_size': 0.8,
        'distribution': 'lognormal',
        'dilution_alpha': 5.0,
        'dilution_beta': 5.0,
        'lod_percentile': 0.1,
        'lod_handling': 'substitute'
    }
    
    # Generate a sample dataset
    print("Generating sample dataset...")
    dataset = generate_dataset(**params)
    
    # Visualize dilution effect
    print("Visualizing dilution effect...")
    fig1 = plot_dilution_effect(dataset)
    fig1.savefig('demo_dilution_effect.png')
    
    # Compare normalization methods
    print("Comparing normalization methods...")
    fig2 = plot_normalization_comparison(dataset)
    fig2.savefig('demo_normalization_comparison.png')
    
    # Run a single simulation with different normalization methods
    print("Running analysis with different normalization methods...")
    normalization_methods = ['none', 'total_sum', 'pqn', 'clr', 'ilr', 'reference', 'quantile']
    results = run_single_simulation(params, normalization_methods)
    
    # Print summary of results
    print("\nSummary of results:")
    print("\nUnivariate analysis metrics:")
    for method, metrics in results['univariate'].items():
        print(f"  {method}: Power = {metrics['power']:.3f}, Type I Error = {metrics['type_i_error']:.3f}")
    
    print("\nCorrelation recovery metrics:")
    for method, metrics in results['correlation'].items():
        print(f"  {method}: Frobenius Norm = {metrics['frobenius_norm']:.3f}, Rank Correlation = {metrics['rank_correlation']:.3f}")
    
    print("\nPCA metrics:")
    for method, metrics in results['pca'].items():
        print(f"  {method}: RV Coefficient = {metrics['rv_coefficient']:.3f}")
    
    print("\nClustering metrics:")
    for method, metrics in results['clustering'].items():
        print(f"  {method}: Adjusted Rand Index = {metrics['adjusted_rand_index']:.3f}")
    
    print("\nClassification metrics:")
    for method, metrics in results['classification'].items():
        print(f"  {method}: Accuracy = {metrics['accuracy']:.3f}, AUC = {metrics.get('auc_roc', 'N/A')}")
    
    return results


def run_comprehensive_simulation():
    """
    Run a comprehensive Monte Carlo simulation with multiple parameter combinations.
    """
    print("Setting up comprehensive simulation...")
    
    # Define parameter grid
    param_grid = {
        'n_subjects': [50, 100, 200],
        'n_biomarkers': [5, 20, 50],
        'n_groups': [2, 3],
        'correlation_type': ['none', 'moderate', 'block'],
        'effect_size': [0.3, 0.8],
        'distribution': ['normal', 'lognormal'],
        'dilution_alpha': [8.0, 2.0],  # 8.0 = mild dilution, 2.0 = severe dilution
        'dilution_beta': [2.0, 8.0],   # pair with above respectively
        'lod_percentile': [0.05, 0.25],
        'lod_handling': ['substitute']
    }
    
    # Simplify for testing - uncomment to run full simulation
    # reduced_param_grid = {
    #     'n_subjects': [100],
    #     'n_biomarkers': [10],
    #     'n_groups': [2],
    #     'correlation_type': ['moderate'],
    #     'effect_size': [0.5],
    #     'distribution': ['lognormal'],
    #     'dilution_alpha': [5.0],
    #     'dilution_beta': [5.0],
    #     'lod_percentile': [0.1],
    #     'lod_handling': ['substitute']
    # }
    
    # Set normalization methods to test
    normalization_methods = ['none', 'total_sum', 'pqn', 'clr']
    
    print("Running Monte Carlo simulation...")
    results = run_monte_carlo_simulation(
        param_grid=param_grid,
        n_replications=10,  # Use higher value for final simulation (e.g., 100)
        normalization_methods=normalization_methods,
        n_processes=4,  # Adjust based on available CPU cores
        output_dir='simulation_results'
    )
    
    print(f"Simulation complete. Results saved to 'simulation_results' directory.")
    return results


def analyze_simulation_results(results_file):
    """
    Analyze and visualize results from a previously run simulation.
    """
    print(f"Analyzing results from {results_file}...")
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} simulation results.")
    
    # Convert to DataFrame for easier analysis
    rows = []
    for res in results:
        # Extract parameters
        params = res['params']
        
        # Process each normalization method
        for method in res['univariate'].keys():
            row = {
                'n_subjects': params['n_subjects'],
                'n_biomarkers': params['n_biomarkers'],
                'n_groups': params['n_groups'],
                'correlation_type': params['correlation_type'],
                'effect_size': params['effect_size'],
                'distribution': params['distribution'],
                'dilution_alpha': params['dilution_alpha'],
                'dilution_beta': params['dilution_beta'],
                'lod_percentile': params['lod_percentile'],
                'normalization': method,
                # Univariate metrics
                'power': res['univariate'][method]['power'],
                'type_i_error': res['univariate'][method]['type_i_error'],
                # Correlation metrics
                'corr_frobenius': res['correlation'][method]['frobenius_norm'],
                'corr_rank_correlation': res['correlation'][method]['rank_correlation'],
                # PCA metrics
                'pca_rv_coefficient': res['pca'][method]['rv_coefficient'],
                # Clustering metrics
                'clustering_ari': res['clustering'][method]['adjusted_rand_index'],
                # Classification metrics
                'classification_accuracy': res['classification'][method]['accuracy'],
                'classification_auc': res['classification'][method].get('auc_roc', np.nan)
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save processed results
    output_dir = Path('simulation_results')
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / 'processed_results.csv', index=False)
    
    # Generate summary visualizations
    
    # 1. Boxplot of metrics by normalization method
    plt.figure(figsize=(12, 8))
    metrics = ['power', 'corr_rank_correlation', 'pca_rv_coefficient', 'clustering_ari', 'classification_accuracy']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='normalization', y=metric, data=df)
        plt.title(f'Distribution of {metric}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_normalization.png')
    
    # 2. Heatmap of metrics by dilution severity and normalization
    # Create dilution severity categories
    df['dilution_severity'] = 'Moderate'
    df.loc[(df['dilution_alpha'] > 5) & (df['dilution_beta'] < 5), 'dilution_severity'] = 'Mild'
    df.loc[(df['dilution_alpha'] < 5) & (df['dilution_beta'] > 5), 'dilution_severity'] = 'Severe'
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        pivot = df.pivot_table(
            index='dilution_severity',
            columns='normalization',
            values=metric,
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title(f'Average {metric} by Dilution Severity and Normalization')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png')
    
    # 3. Performance by number of biomarkers
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        for norm in df['normalization'].unique():
            subset = df[df['normalization'] == norm]
            means = subset.groupby('n_biomarkers')[metric].mean()
            plt.plot(means.index, means.values, 'o-', label=norm)
        
        plt.xlabel('Number of Biomarkers')
        plt.ylabel(metric)
        plt.title(f'{metric} vs Number of Biomarkers')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_n_biomarkers.png')
    
    print(f"Analysis complete. Visualizations saved to {output_dir}")
    return df


def run_enhanced_simulation(config_file: str = None):
    """
    Run an enhanced simulation with all new features.
    """
    print("Running enhanced simulation with new features...")

    # Setup experiment tracking
    tracker, config, repro = create_experiment(
        name="enhanced_simulation",
        config=config_file,
        seed=42
    )

    try:
        # Get simulation parameters
        params = config.get_simulation_params()
        print(f"Simulation parameters: {params}")

        # Generate dataset with new dilution model options
        dataset = generate_dataset(**params)

        tracker.log_metric('n_samples', len(dataset['y']))
        tracker.log_metric('n_biomarkers', dataset['X_true'].shape[1])

        # Test new normalization methods
        print("\nTesting normalization methods...")
        norm_methods = ['none', 'total_sum', 'pqn', 'clr', 'median', 'vsn', 'quantile']

        results = run_single_simulation(params, norm_methods)

        # Log metrics for each normalization method
        for method in norm_methods:
            if method in results['univariate']:
                tracker.log_metric(f'{method}_power', results['univariate'][method]['power'])
                tracker.log_metric(f'{method}_type_i_error', results['univariate'][method]['type_i_error'])

        # Test enhanced statistical analysis
        print("\nRunning enhanced statistical analysis...")
        X_obs = dataset['X_obs']
        y = dataset['y']

        # Multiple testing correction
        p_values, _ = analyze_univariate(X_obs, y, 't_test')
        p_adjusted, significant = multiple_testing_correction(p_values, 'fdr_bh')
        print(f"  Significant biomarkers (FDR < 0.05): {np.sum(significant)}")

        # Effect sizes
        effect_sizes = calculate_effect_size(X_obs, y, 'cohens_d')
        print(f"  Mean absolute effect size: {np.mean(np.abs(effect_sizes)):.3f}")

        # Bootstrap confidence intervals
        es_point, es_lower, es_upper = bootstrap_confidence_interval(
            X_obs, y, lambda x, y: calculate_effect_size(x, y, 'cohens_d'),
            n_bootstrap=100
        )
        print(f"  Effect size 95% CI width: {np.mean(es_upper - es_lower):.3f}")

        # Test advanced ML methods
        print("\nTesting advanced ML methods...")
        ml_methods = ['logistic', 'random_forest', 'gradient_boosting']

        for ml_method in ml_methods:
            cv_results = cross_validate_classification(X_obs, y, ml_method, n_folds=5)
            accuracy = cv_results['summary'].get('accuracy_mean', 0)
            print(f"  {ml_method}: Accuracy = {accuracy:.3f}")
            tracker.log_metric(f'ml_{ml_method}_accuracy', accuracy)

        # Test feature selection
        print("\nTesting feature selection...")
        selected_features, importances = feature_selection(X_obs, y, method='random_forest', n_features=5)
        print(f"  Selected features: {selected_features}")

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Volcano plot
        enhanced_results = analyze_univariate_enhanced(X_obs, y)
        if 'fold_changes' in enhanced_results:
            fig_volcano = plot_volcano(
                enhanced_results['fold_changes'],
                enhanced_results['p_values'],
                title="Volcano Plot - Enhanced Analysis"
            )
            tracker.save_artifact('volcano_plot', fig_volcano, 'figure')
            plt.close(fig_volcano)

        # 3D PCA
        fig_3d = plot_pca_3d(X_obs, y, dataset['dilution_factors'])
        tracker.save_artifact('pca_3d', fig_3d, 'figure')
        plt.close(fig_3d)

        # Forest plot
        fig_forest = plot_forest(effect_sizes, es_lower, es_upper)
        tracker.save_artifact('forest_plot', fig_forest, 'figure')
        plt.close(fig_forest)

        # Save results
        tracker.save_artifact('results', results, 'pickle')

        # Finish experiment
        tracker.finish('completed')
        print(f"\nExperiment completed! Results saved to: {tracker.get_output_dir()}")

        return results

    except Exception as e:
        tracker.finish('failed')
        print(f"Experiment failed: {e}")
        raise


def run_dilution_model_comparison():
    """
    Compare different dilution models.
    """
    print("Comparing dilution models...")

    n_subjects = 200
    dilution_models = {
        'beta_mild': generate_dilution_factors(n_subjects, 8.0, 2.0, 'beta'),
        'beta_severe': generate_dilution_factors(n_subjects, 2.0, 8.0, 'beta'),
        'bimodal': generate_dilution_factors(n_subjects, 5.0, 5.0, 'bimodal'),
        'mixture': generate_dilution_factors(n_subjects, 5.0, 5.0, 'mixture'),
        'uniform': generate_dilution_factors(n_subjects, 5.0, 5.0, 'uniform'),
    }

    # Visualize distributions
    fig = plot_dilution_distribution_comparison(dilution_models)
    fig.savefig('dilution_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("Dilution model comparison saved to 'dilution_model_comparison.png'")

    # Statistics for each model
    print("\nDilution Model Statistics:")
    print("-" * 60)
    for name, factors in dilution_models.items():
        print(f"{name:15s}: mean={np.mean(factors):.3f}, std={np.std(factors):.3f}, "
              f"min={np.min(factors):.3f}, max={np.max(factors):.3f}")

    return dilution_models


def run_power_analysis_simulation():
    """
    Run power analysis for different scenarios.
    """
    print("Running power analysis...")

    effect_sizes = np.array([0.2, 0.5, 0.8, 1.0, 1.5])
    sample_sizes = np.arange(10, 201, 10)

    # Plot power curves
    fig = plot_power_curve(effect_sizes, sample_sizes)
    fig.savefig('power_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("Power analysis saved to 'power_analysis.png'")

    # Sample size recommendations
    print("\nSample size recommendations for 80% power:")
    print("-" * 40)
    for es in effect_sizes:
        n_required = sample_size_estimation(es, power=0.8)
        print(f"Effect size {es:.1f}: n = {n_required} per group")

    return effect_sizes, sample_sizes


def main():
    parser = argparse.ArgumentParser(description='Biomarker Dilution Monte Carlo Simulation')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'full', 'analyze', 'enhanced', 'dilution', 'power'],
                        help='Simulation mode')
    parser.add_argument('--results', type=str, help='Results file for analysis')
    parser.add_argument('--config', type=str, help='Configuration file (YAML or JSON)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    if args.mode == 'demo':
        results = run_demo_simulation()
    elif args.mode == 'full':
        results = run_comprehensive_simulation()
    elif args.mode == 'analyze':
        if args.results is None:
            print("Error: --results file path is required for analyze mode")
            return
        df = analyze_simulation_results(args.results)
    elif args.mode == 'enhanced':
        results = run_enhanced_simulation(args.config)
    elif args.mode == 'dilution':
        run_dilution_model_comparison()
    elif args.mode == 'power':
        run_power_analysis_simulation()
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
