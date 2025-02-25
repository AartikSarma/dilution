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

# Import the simulation module
# Assuming the previous code is saved in a module named 'biomarker_dilution_sim'
from biomarker_dilution_sim import *

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


def main():
    parser = argparse.ArgumentParser(description='Biomarker Dilution Monte Carlo Simulation')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'full', 'analyze'],
                        help='Simulation mode: demo, full, or analyze')
    parser.add_argument('--results', type=str, help='Results file for analysis')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        results = run_demo_simulation()
    elif args.mode == 'full':
        results = run_comprehensive_simulation()
    elif args.mode == 'analyze':
        if args.results is None:
            print("Error: --results file path is required for analyze mode")
            return
        df = analyze_simulation_results(args.results)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
