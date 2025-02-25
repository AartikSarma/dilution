import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics, decomposition, cluster, preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import itertools
from tqdm import tqdm
import multiprocessing as mp
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

#-------------------------------------------------------
# Data Generation Functions
#-------------------------------------------------------

def generate_correlation_matrix(
    n_biomarkers: int,
    correlation_type: str = 'moderate',
    block_size: Optional[int] = None
) -> np.ndarray:
    """
    Generate a correlation matrix with specified structure.
    
    Parameters:
    -----------
    n_biomarkers : int
        Number of biomarkers
    correlation_type : str
        Type of correlation: 'none', 'low', 'moderate', 'high', 'block'
    block_size : int, optional
        Size of blocks if correlation_type is 'block'
        
    Returns:
    --------
    np.ndarray
        Correlation matrix (symmetric, positive definite)
    """
    if correlation_type == 'none':
        # Identity matrix (no correlation)
        return np.eye(n_biomarkers)
    
    if correlation_type == 'block' and block_size is not None:
        # Block correlation structure
        n_blocks = int(np.ceil(n_biomarkers / block_size))
        corr_matrix = np.zeros((n_biomarkers, n_biomarkers))
        
        for i in range(n_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, n_biomarkers)
            block_size_actual = end - start
            
            # Create a block with high correlation
            block = np.ones((block_size_actual, block_size_actual)) * 0.7
            np.fill_diagonal(block, 1.0)
            
            corr_matrix[start:end, start:end] = block
        
        # Ensure the matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(corr_matrix))
        if min_eig < 0:
            corr_matrix += (-min_eig + 0.01) * np.eye(n_biomarkers)
            
        # Normalize to ensure diagonal is 1
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)
        
        return corr_matrix
    
    # For low, moderate, high correlations
    corr_values = {
        'low': 0.2,
        'moderate': 0.5,
        'high': 0.8
    }
    
    # Create matrix with same correlation value off-diagonal
    corr_val = corr_values.get(correlation_type, 0.5)
    corr_matrix = np.ones((n_biomarkers, n_biomarkers)) * corr_val
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix


def generate_covariance_matrix(
    correlation_matrix: np.ndarray,
    variances: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert correlation matrix to covariance matrix with specified variances.
    """
    n = correlation_matrix.shape[0]
    
    if variances is None:
        # Default to unit variance
        variances = np.ones(n)
    
    # Convert correlation to covariance
    std_devs = np.sqrt(variances)
    cov_matrix = correlation_matrix * np.outer(std_devs, std_devs)
    
    return cov_matrix


def generate_group_means(
    n_biomarkers: int,
    n_groups: int,
    effect_sizes: Union[float, List[float], np.ndarray] = 1.0,
    biomarker_scales: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate mean values for each group and biomarker with specified effect sizes.
    """
    if biomarker_scales is None:
        # Default biomarker scales (can vary by orders of magnitude)
        biomarker_scales = 10 ** np.random.uniform(-1, 3, size=n_biomarkers)
    
    # Standardize effect_sizes to array
    if isinstance(effect_sizes, (int, float)):
        effect_sizes = np.ones(n_biomarkers) * effect_sizes
    
    # Initialize means array
    group_means = np.zeros((n_groups, n_biomarkers))
    
    # First group is baseline
    group_means[0, :] = biomarker_scales
    
    # Generate means for other groups with effect sizes
    for g in range(1, n_groups):
        # Group-specific factor (to create different patterns between groups)
        group_factor = np.random.choice([-1, 1], size=n_biomarkers)
        
        # Some biomarkers may not differ between groups
        differential_markers = np.random.choice(
            [0, 1], size=n_biomarkers, p=[0.3, 0.7]
        )
        
        # Set means with specified effect sizes
        effect = effect_sizes * differential_markers * group_factor
        group_means[g, :] = group_means[0, :] * (1 + effect)
    
    return group_means


def generate_true_biomarker_data(
    n_subjects: int,
    n_biomarkers: int,
    n_groups: int,
    group_means: np.ndarray,
    cov_matrix: np.ndarray,
    distribution: str = 'normal',
    group_sizes: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate true biomarker concentrations for multiple groups.
    """
    # Determine group sizes
    if group_sizes is None:
        # Equal size groups by default
        group_sizes = [n_subjects // n_groups] * n_groups
        group_sizes[0] += n_subjects - sum(group_sizes)  # Add remainder
    
    assert sum(group_sizes) == n_subjects, "Group sizes must sum to n_subjects"
    
    # Initialize data arrays
    X = np.zeros((n_subjects, n_biomarkers))
    y = np.zeros(n_subjects, dtype=int)
    
    # Generate data for each group
    start_idx = 0
    for g in range(n_groups):
        end_idx = start_idx + group_sizes[g]
        
        if distribution == 'normal':
            # Generate from multivariate normal
            X[start_idx:end_idx, :] = np.random.multivariate_normal(
                mean=group_means[g, :],
                cov=cov_matrix,
                size=group_sizes[g]
            )
            # Ensure non-negative values
            X[start_idx:end_idx, :] = np.maximum(X[start_idx:end_idx, :], 0.001)
            
        elif distribution == 'lognormal':
            # Adjust parameters for log-normal
            log_means = np.log(group_means[g, :]) - 0.5 * np.diag(cov_matrix)
            
            # Generate on log scale
            log_X = np.random.multivariate_normal(
                mean=log_means,
                cov=cov_matrix,
                size=group_sizes[g]
            )
            
            # Transform to original scale
            X[start_idx:end_idx, :] = np.exp(log_X)
            
        elif distribution == 'mixed':
            # Mixed distributions: some normal, some log-normal
            is_lognormal = np.random.choice([True, False], size=n_biomarkers)
            
            # Generate multivariate normal data first
            data = np.random.multivariate_normal(
                mean=group_means[g, :],
                cov=cov_matrix,
                size=group_sizes[g]
            )
            
            # Transform log-normal variables
            for j in range(n_biomarkers):
                if is_lognormal[j]:
                    log_mean = np.log(group_means[g, j]) - 0.5 * cov_matrix[j, j]
                    log_data = np.random.normal(
                        loc=log_mean,
                        scale=np.sqrt(cov_matrix[j, j]),
                        size=group_sizes[g]
                    )
                    data[:, j] = np.exp(log_data)
                    
            X[start_idx:end_idx, :] = np.maximum(data, 0.001)
        
        # Assign group labels
        y[start_idx:end_idx] = g
        
        # Update index
        start_idx = end_idx
    
    return X, y


def generate_dilution_factors(
    n_subjects: int,
    alpha: float = 5.0,
    beta: float = 5.0
) -> np.ndarray:
    """
    Generate dilution factors from a Beta distribution.
    """
    return np.random.beta(alpha, beta, size=n_subjects)


def apply_dilution(
    true_concentrations: np.ndarray,
    dilution_factors: np.ndarray
) -> np.ndarray:
    """
    Apply dilution to true biomarker concentrations.
    """
    # Apply dilution factors (reshape for broadcasting)
    return true_concentrations * dilution_factors.reshape(-1, 1)


def apply_detection_limits(
    concentrations: np.ndarray,
    lods: np.ndarray,
    handling_method: str = 'substitute'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply detection limits to biomarker concentrations.
    """
    n_subjects, n_biomarkers = concentrations.shape
    
    # Initialize censoring mask
    censored = np.zeros_like(concentrations, dtype=bool)
    
    # Create output array
    output = concentrations.copy()
    
    # Apply LOD for each biomarker
    for j in range(n_biomarkers):
        below_lod = concentrations[:, j] < lods[j]
        censored[:, j] = below_lod
        
        if handling_method == 'substitute':
            # Replace with LOD/√2
            output[below_lod, j] = lods[j] / np.sqrt(2)
        
        elif handling_method == 'zero':
            # Replace with zero
            output[below_lod, j] = 0
            
        elif handling_method == 'min':
            # Replace with minimum observed value
            if not all(below_lod):
                min_observed = np.min(concentrations[~below_lod, j])
                output[below_lod, j] = min_observed
            else:
                output[below_lod, j] = lods[j] / 2
    
    return output, censored


def generate_limits_of_detection(
    true_concentrations: np.ndarray,
    percentiles: Union[float, List[float], np.ndarray] = 0.1
) -> np.ndarray:
    """
    Generate limits of detection based on percentiles of true concentrations.
    
    Parameters:
    -----------
    true_concentrations : np.ndarray
        True biomarker concentrations
    percentiles : float or array-like
        Percentile(s) to use for LOD (between 0 and 1)
        
    Returns:
    --------
    np.ndarray
        Limits of detection for each biomarker
    """
    n_biomarkers = true_concentrations.shape[1]
    
    # Convert to array if scalar
    if isinstance(percentiles, (int, float)):
        percentiles = np.ones(n_biomarkers) * percentiles
    
    # Calculate LODs for each biomarker
    lods = np.zeros(n_biomarkers)
    for j in range(n_biomarkers):
        lods[j] = np.percentile(true_concentrations[:, j], percentiles[j] * 100)
    
    return lods


def generate_dataset(
    n_subjects: int = 100,
    n_biomarkers: int = 10,
    n_groups: int = 2,
    correlation_type: str = 'moderate',
    effect_size: float = 0.5,
    distribution: str = 'lognormal',
    dilution_alpha: float = 5.0,
    dilution_beta: float = 5.0,
    lod_percentile: float = 0.1,
    lod_handling: str = 'substitute',
    block_size: Optional[int] = None,
    group_sizes: Optional[List[int]] = None,
    biomarker_scales: Optional[np.ndarray] = None
) -> Dict:
    """
    Generate a complete dataset with true and observed biomarker concentrations.
    
    Returns:
    --------
    Dict containing:
        - X_true: True biomarker concentrations
        - X_obs: Observed (diluted) biomarker concentrations
        - y: Group labels
        - dilution_factors: Dilution factors
        - censored: Boolean mask for censored values
        - lods: Limits of detection
        - params: Input parameters
    """
    # Store parameters
    params = {
        'n_subjects': n_subjects,
        'n_biomarkers': n_biomarkers,
        'n_groups': n_groups,
        'correlation_type': correlation_type,
        'effect_size': effect_size,
        'distribution': distribution,
        'dilution_alpha': dilution_alpha,
        'dilution_beta': dilution_beta,
        'lod_percentile': lod_percentile
    }
    
    # Generate correlation and covariance matrices
    corr_matrix = generate_correlation_matrix(
        n_biomarkers=n_biomarkers,
        correlation_type=correlation_type,
        block_size=block_size
    )
    
    # Define variances based on distribution
    if distribution == 'lognormal':
        # Higher variance for lognormal to get similar spread in original scale
        variances = np.ones(n_biomarkers) * 0.5
    else:
        variances = np.ones(n_biomarkers)
    
    cov_matrix = generate_covariance_matrix(corr_matrix, variances)
    
    # Generate group means
    group_means = generate_group_means(
        n_biomarkers=n_biomarkers,
        n_groups=n_groups,
        effect_sizes=effect_size,
        biomarker_scales=biomarker_scales
    )
    
    # Generate true biomarker data
    X_true, y = generate_true_biomarker_data(
        n_subjects=n_subjects,
        n_biomarkers=n_biomarkers,
        n_groups=n_groups,
        group_means=group_means,
        cov_matrix=cov_matrix,
        distribution=distribution,
        group_sizes=group_sizes
    )
    
    # Generate dilution factors
    dilution_factors = generate_dilution_factors(
        n_subjects=n_subjects,
        alpha=dilution_alpha,
        beta=dilution_beta
    )
    
    # Apply dilution
    X_diluted = apply_dilution(X_true, dilution_factors)
    
    # Generate limits of detection
    lods = generate_limits_of_detection(X_diluted, lod_percentile)
    
    # Apply detection limits
    X_obs, censored = apply_detection_limits(
        concentrations=X_diluted,
        lods=lods,
        handling_method=lod_handling
    )
    
    # Return all relevant data and parameters
    return {
        'X_true': X_true,
        'X_obs': X_obs,
        'y': y,
        'dilution_factors': dilution_factors,
        'censored': censored,
        'lods': lods,
        'corr_matrix': corr_matrix,
        'group_means': group_means,
        'params': params
    }

#-------------------------------------------------------
# Normalization Methods
#-------------------------------------------------------

def normalize_total_sum(X: np.ndarray) -> np.ndarray:
    """
    Total sum normalization (convert to relative abundances).
    """
    row_sums = X.sum(axis=1, keepdims=True)
    # Handle zero sums (unlikely but possible)
    row_sums[row_sums == 0] = 1
    return X / row_sums


def normalize_reference_biomarker(
    X: np.ndarray,
    reference_idx: int = 0
) -> np.ndarray:
    """
    Normalize using a reference biomarker.
    """
    reference_values = X[:, reference_idx].reshape(-1, 1)
    # Handle zero reference values
    reference_values[reference_values == 0] = np.min(reference_values[reference_values > 0])
    return X / reference_values


def normalize_probabilistic_quotient(X: np.ndarray) -> np.ndarray:
    """
    Probabilistic Quotient Normalization (PQN).
    """
    # Calculate reference spectrum (median)
    reference = np.median(X, axis=0)
    
    # Avoid zeros in reference
    reference[reference == 0] = np.min(reference[reference > 0])
    
    # Calculate quotients
    quotients = X / reference
    
    # Calculate normalization factor (median of quotients for each sample)
    norm_factors = np.median(quotients, axis=1).reshape(-1, 1)
    
    # Normalize
    return X / norm_factors


def centered_log_ratio(X: np.ndarray, pseudo_count: float = 1e-5) -> np.ndarray:
    """
    Centered Log-Ratio (CLR) transformation.
    """
    # Add small value to avoid log(0)
    X_positive = X + pseudo_count
    
    # Calculate geometric mean of each sample
    geo_means = stats.gmean(X_positive, axis=1).reshape(-1, 1)
    
    # CLR transformation
    return np.log(X_positive / geo_means)


def isometric_log_ratio(X: np.ndarray, pseudo_count: float = 1e-5) -> np.ndarray:
    """
    Isometric Log-Ratio (ILR) transformation.
    
    Note: This implementation preserves the original dimensionality for compatibility
    with the analysis pipeline, although technically ILR reduces dimensionality.
    """
    n_samples, n_features = X.shape
    
    # Add small value to avoid log(0)
    X_positive = X + pseudo_count
    
    # Convert to proportions
    X_comp = X_positive / X_positive.sum(axis=1, keepdims=True)
    
    # We need n_features - 1 ILR coordinates
    ilr_data = np.zeros((n_samples, n_features - 1))
    
    # Calculate ILR coordinates
    for j in range(n_features - 1):
        # Compute the ILR coordinate
        ilr_data[:, j] = np.sqrt((j + 1) / (j + 2)) * np.log(
            stats.gmean(X_comp[:, j+1:], axis=1) / X_comp[:, j]
        )
    
    # Pad with zeros to maintain original dimensionality for compatibility
    padded_ilr_data = np.zeros((n_samples, n_features))
    padded_ilr_data[:, :-1] = ilr_data
    
    return padded_ilr_data


def normalize_quantile(X: np.ndarray) -> np.ndarray:
    """
    Quantile normalization forces the distributions to be identical.
    """
    # Get ranks for each column (sample)
    ranks = np.zeros_like(X)
    for j in range(X.shape[1]):
        ranks[:, j] = stats.rankdata(X[:, j])
    
    # Get means for each rank (across all columns)
    sorted_X = np.sort(X, axis=0)
    means = np.mean(sorted_X, axis=1)
    
    # Create normalized data
    X_norm = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            rank = int(ranks[i, j]) - 1  # ranks start at 1
            X_norm[i, j] = means[rank]
    
    return X_norm


def normalize_data(
    X: np.ndarray,
    method: str = 'none',
    reference_idx: Optional[int] = None
) -> np.ndarray:
    """
    Apply normalization method to the data.
    """
    # Handle missing or zero values for some methods
    X_clean = X.copy()
    X_clean[X_clean <= 0] = np.min(X_clean[X_clean > 0]) / 10
    
    if method == 'none':
        return X
    elif method == 'total_sum':
        return normalize_total_sum(X_clean)
    elif method == 'reference':
        if reference_idx is None:
            reference_idx = 0
        return normalize_reference_biomarker(X_clean, reference_idx)
    elif method == 'pqn':
        return normalize_probabilistic_quotient(X_clean)
    elif method == 'clr':
        return centered_log_ratio(X_clean)
    elif method == 'ilr':
        return isometric_log_ratio(X_clean)
    elif method == 'quantile':
        return normalize_quantile(X_clean)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

#-------------------------------------------------------
# Analysis Methods
#-------------------------------------------------------

def analyze_univariate(
    X: np.ndarray,
    y: np.ndarray,
    test_type: str = 't_test'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform univariate statistical tests to compare groups.
    
    Returns:
    --------
    Tuple of (p_values, test_statistics)
    """
    n_biomarkers = X.shape[1]
    unique_groups = np.unique(y)
    n_groups = len(unique_groups)
    
    p_values = np.zeros(n_biomarkers)
    test_stats = np.zeros(n_biomarkers)
    
    for j in range(n_biomarkers):
        if test_type == 't_test':
            if n_groups == 2:
                # Two-sample t-test
                group0 = X[y == unique_groups[0], j]
                group1 = X[y == unique_groups[1], j]
                t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)
                test_stats[j] = t_stat
                p_values[j] = p_val
            else:
                # One-way ANOVA
                groups = [X[y == group, j] for group in unique_groups]
                f_stat, p_val = stats.f_oneway(*groups)
                test_stats[j] = f_stat
                p_values[j] = p_val
        
        elif test_type == 'wilcoxon':
            if n_groups == 2:
                # Wilcoxon rank-sum test
                group0 = X[y == unique_groups[0], j]
                group1 = X[y == unique_groups[1], j]
                u_stat, p_val = stats.mannwhitneyu(group0, group1)
                test_stats[j] = u_stat
                p_values[j] = p_val
            else:
                # Kruskal-Wallis test
                groups = [X[y == group, j] for group in unique_groups]
                h_stat, p_val = stats.kruskal(*groups)
                test_stats[j] = h_stat
                p_values[j] = p_val
    
    return p_values, test_stats


def analyze_correlation(X: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrix between biomarkers.
    """
    return np.corrcoef(X.T)


def analyze_pca(X: np.ndarray, n_components: int = 2) -> Tuple:
    """
    Perform Principal Component Analysis.
    
    Returns:
    --------
    Tuple of (transformed data, explained variance ratios, components)
    """
    # Standardize data
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = decomposition.PCA(n_components=min(n_components, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca.explained_variance_ratio_, pca.components_


def analyze_clustering(
    X: np.ndarray,
    n_clusters: int,
    method: str = 'kmeans'
) -> np.ndarray:
    """
    Perform clustering analysis.
    
    Returns:
    --------
    np.ndarray
        Cluster assignments
    """
    # Standardize data
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'kmeans':
        # K-means clustering
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
    
    elif method == 'hierarchical':
        # Hierarchical clustering
        agg = cluster.AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X_scaled)
    
    elif method == 'dbscan':
        # DBSCAN - automatically determines number of clusters
        dbscan = cluster.DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
    
    return labels


def analyze_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str = 'logistic'
) -> np.ndarray:
    """
    Train classifier and make predictions.
    
    Returns:
    --------
    np.ndarray
        Predicted class probabilities for each sample
    """
    from sklearn import linear_model, ensemble, svm
    
    # Standardize data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if method == 'logistic':
        # Logistic regression
        clf = linear_model.LogisticRegression(max_iter=1000, random_state=42)
    
    elif method == 'random_forest':
        # Random forest
        clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
    
    elif method == 'svm':
        # Support vector machine
        clf = svm.SVC(probability=True, random_state=42)
    
    # Train model
    clf.fit(X_train_scaled, y_train)
    
    # Predict probabilities
    return clf.predict_proba(X_test_scaled)

#-------------------------------------------------------
# Evaluation Metrics
#-------------------------------------------------------

def evaluate_univariate(
    p_values_true: np.ndarray,
    p_values_obs: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Evaluate performance of univariate tests.
    """
    # Handle potential dimension mismatch
    if p_values_true.shape != p_values_obs.shape:
        # Use the minimum length to truncate the longer array
        min_len = min(len(p_values_true), len(p_values_obs))
        p_values_true = p_values_true[:min_len]
        p_values_obs = p_values_obs[:min_len]
        
        # Print a warning
        print(f"Warning: p-value arrays had different shapes. Truncated to {min_len} elements.")
    
    # Determine significant biomarkers
    sig_true = p_values_true < alpha
    sig_obs = p_values_obs < alpha
    
    # Calculate metrics
    true_positives = np.sum(sig_true & sig_obs)
    false_positives = np.sum(~sig_true & sig_obs)
    true_negatives = np.sum(~sig_true & ~sig_obs)
    false_negatives = np.sum(sig_true & ~sig_obs)
    
    # Calculate rates
    power = true_positives / max(1, np.sum(sig_true))
    type_i_error = false_positives / max(1, np.sum(~sig_true))
    
    # Calculate rank correlation of p-values
    rank_corr = stats.spearmanr(p_values_true, p_values_obs)[0]
    
    # Calculate -log10 error
    log_error = np.mean(np.abs(np.log10(p_values_true + 1e-10) - np.log10(p_values_obs + 1e-10)))
    
    return {
        'power': power,
        'type_i_error': type_i_error,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'rank_correlation': rank_corr,
        'log_p_value_error': log_error
    }

def evaluate_correlation(
    corr_matrix_true: np.ndarray,
    corr_matrix_obs: np.ndarray,
    threshold: float = 0.3
) -> Dict:
    """
    Evaluate correlation matrix recovery.
    """
    # Handle potential dimension mismatch
    if corr_matrix_true.shape != corr_matrix_obs.shape:
        min_dim = min(corr_matrix_true.shape[0], corr_matrix_obs.shape[0])
        corr_matrix_true = corr_matrix_true[:min_dim, :min_dim]
        corr_matrix_obs = corr_matrix_obs[:min_dim, :min_dim]
        
        print(f"Warning: correlation matrices had different shapes. Truncated to {min_dim}x{min_dim}.")
    
    # Mask for off-diagonal elements
    off_diag = ~np.eye(corr_matrix_true.shape[0], dtype=bool)
    
    # Calculate Frobenius norm of difference
    frob_norm = np.linalg.norm(corr_matrix_true - corr_matrix_obs, 'fro')
    
    # Calculate element-wise error
    element_error = np.mean(np.abs(corr_matrix_true[off_diag] - corr_matrix_obs[off_diag]))
    
    # Calculate false correlations
    sig_true = np.abs(corr_matrix_true) > threshold
    sig_obs = np.abs(corr_matrix_obs) > threshold
    
    # Only consider off-diagonal elements
    sig_true = sig_true & off_diag
    sig_obs = sig_obs & off_diag
    
    # Calculate metrics
    true_positives = np.sum(sig_true & sig_obs)
    false_positives = np.sum(~sig_true & sig_obs)
    true_negatives = np.sum(~sig_true & ~sig_obs)
    false_negatives = np.sum(sig_true & ~sig_obs)
    
    # Calculate rates
    sensitivity = true_positives / max(1, np.sum(sig_true))
    specificity = true_negatives / max(1, np.sum(~sig_true))
    precision = true_positives / max(1, true_positives + false_positives)
    
    # Calculate rank correlation of correlation values
    rank_corr = stats.spearmanr(corr_matrix_true[off_diag], corr_matrix_obs[off_diag])[0]
    
    return {
        'frobenius_norm': frob_norm,
        'element_error': element_error,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'rank_correlation': rank_corr
    }

def evaluate_pca(
    X_pca_true: np.ndarray,
    X_pca_obs: np.ndarray,
    var_explained_true: np.ndarray,
    var_explained_obs: np.ndarray,
    components_true: np.ndarray,
    components_obs: np.ndarray
) -> Dict:
    """
    Evaluate PCA recovery.
    """
    # Calculate variance explained by first component (dilution effect)
    var_pc1_true = var_explained_true[0]
    var_pc1_obs = var_explained_obs[0]
    
    # Calculate absolute loadings correlation (sign may flip)
    loadings_corr = np.zeros(min(components_true.shape[0], components_obs.shape[0]))
    for i in range(len(loadings_corr)):
        loadings_corr[i] = np.abs(np.corrcoef(components_true[i, :], components_obs[i, :])[0, 1])
    
    # Calculate RV coefficient (measure of matrix similarity)
    def rv_coefficient(X, Y):
        RX = X @ X.T
        RY = Y @ Y.T
        # Frobenius inner products
        frob_inner_XY = np.sum(RX * RY)
        frob_inner_XX = np.sum(RX * RX)
        frob_inner_YY = np.sum(RY * RY)
        return frob_inner_XY / np.sqrt(frob_inner_XX * frob_inner_YY)
    
    # Calculate RV coefficient
    rv_coef = rv_coefficient(X_pca_true, X_pca_obs)
    
    return {
        'variance_pc1_true': var_pc1_true,
        'variance_pc1_obs': var_pc1_obs,
        'loadings_correlation': loadings_corr,
        'rv_coefficient': rv_coef
    }


def evaluate_clustering(
    labels_true: np.ndarray,
    labels_obs: np.ndarray
) -> Dict:
    """
    Evaluate clustering performance.
    """
    # Calculate adjusted Rand index
    ari = metrics.adjusted_rand_score(labels_true, labels_obs)
    
    # Calculate normalized mutual information
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_obs)
    
    # Calculate adjusted mutual information
    ami = metrics.adjusted_mutual_info_score(labels_true, labels_obs)
    
    return {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'adjusted_mutual_info': ami
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Dict:
    """
    Evaluate classification performance.
    """
    # Convert to one-hot encoding for multi-class metrics
    n_classes = y_proba.shape[1]
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1
    
    # Calculate metrics for binary case
    if n_classes == 2:
        # Predicted labels
        y_pred = np.argmax(y_proba, axis=1)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        
        # Calculate AUC
        if len(np.unique(y_true)) > 1:
            auc = metrics.roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = np.nan
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc
        }
    
    # Calculate metrics for multi-class case
    else:
        # Predicted labels
        y_pred = np.argmax(y_proba, axis=1)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
        f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
        
        # Calculate AUC (one-vs-rest)
        try:
            auc = metrics.roc_auc_score(y_true_onehot, y_proba, average='macro', multi_class='ovr')
        except:
            auc = np.nan
        
        return {
            'accuracy': accuracy,
            'f1_score_macro': f1_macro,
            'f1_score_weighted': f1_weighted,
            'auc_roc': auc
        }

#-------------------------------------------------------
# Main Simulation Function
#-------------------------------------------------------

def run_single_simulation(
    params: Dict,
    normalization_methods: List[str] = ['none', 'total_sum', 'pqn', 'clr']
) -> Dict:
    """
    Run a single Monte Carlo simulation with specified parameters.
    """
    # Generate dataset
    data = generate_dataset(**params)
    
    X_true = data['X_true']
    X_obs = data['X_obs']
    y = data['y']
    n_groups = len(np.unique(y))
    
    # Initialize results dictionary
    results = {
        'params': params,
        'univariate': {},
        'correlation': {},
        'pca': {},
        'clustering': {},
        'classification': {}
    }
    
    # Calculate metrics on true data
    # Univariate tests
    p_values_true, _ = analyze_univariate(X_true, y, 't_test')
    
    # Correlation
    corr_matrix_true = analyze_correlation(X_true)
    
    # PCA
    X_pca_true, var_explained_true, components_true = analyze_pca(X_true)
    
    # Clustering (if multiple groups)
    if n_groups > 1:
        labels_true = y
    else:
        # Use k-means on true data if no group labels
        labels_true = analyze_clustering(X_true, n_groups)
    
    # Evaluate methods on observed data with different normalizations
    for norm_method in normalization_methods:
        # Apply normalization
        X_norm = normalize_data(X_obs, norm_method)
        
        # Univariate tests
        p_values_norm, _ = analyze_univariate(X_norm, y, 't_test')
        results['univariate'][norm_method] = evaluate_univariate(p_values_true, p_values_norm)
        
        # Correlation
        corr_matrix_norm = analyze_correlation(X_norm)
        results['correlation'][norm_method] = evaluate_correlation(corr_matrix_true, corr_matrix_norm)
        
        # PCA
        X_pca_norm, var_explained_norm, components_norm = analyze_pca(X_norm)
        results['pca'][norm_method] = evaluate_pca(
            X_pca_true, X_pca_norm,
            var_explained_true, var_explained_norm,
            components_true, components_norm
        )
        
        # Clustering
        labels_norm = analyze_clustering(X_norm, n_groups)
        results['clustering'][norm_method] = evaluate_clustering(labels_true, labels_norm)
        
        # Classification (using cross-validation)
        from sklearn.model_selection import StratifiedKFold
        
        # Initialize metrics
        cv_metrics = []
        
        # 5-fold cross-validation
        kf = StratifiedKFold(n_splits=5)
        for train_idx, test_idx in kf.split(X_norm, y):
            X_train, X_test = X_norm[train_idx], X_norm[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train classifier and predict
            y_proba = analyze_classification(X_train, y_train, X_test, 'logistic')
            
            # Evaluate
            metrics_fold = evaluate_classification(y_test, y_proba)
            cv_metrics.append(metrics_fold)
        
        # Average metrics across folds
        metrics_avg = {}
        for key in cv_metrics[0].keys():
            metrics_avg[key] = np.mean([m[key] for m in cv_metrics])
        
        results['classification'][norm_method] = metrics_avg
    
    return results


def run_monte_carlo_simulation(
    param_grid: Dict,
    n_replications: int = 100,
    normalization_methods: List[str] = ['none', 'total_sum', 'pqn', 'clr'],
    n_processes: int = 1,
    output_dir: str = 'simulation_results'
) -> Dict:
    """
    Run Monte Carlo simulation with multiple parameter combinations.
    """
    # Create parameter combinations
    param_keys = param_grid.keys()
    param_values = param_grid.values()
    param_combinations = list(itertools.product(*param_values))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results list
    all_results = []
    
    # Run simulations
    print(f"Running {len(param_combinations)} parameter combinations with {n_replications} replications each")
    
    # Function to run a batch of replications
    def run_replications(params_dict, n_reps):
        results = []
        for _ in range(n_reps):
            result = run_single_simulation(params_dict, normalization_methods)
            results.append(result)
        return results
    
    # Process parameter combinations
    if n_processes > 1:
        # Parallel processing
        with mp.Pool(processes=n_processes) as pool:
            tasks = []
            for combo in param_combinations:
                params_dict = dict(zip(param_keys, combo))
                tasks.append(pool.apply_async(run_replications, (params_dict, n_replications)))
            
            # Collect results
            for i, task in enumerate(tqdm(tasks)):
                result_batch = task.get()
                all_results.extend(result_batch)
                
                # Save intermediate results periodically
                if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{output_dir}/sim_results_intermediate_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(all_results, f)
    else:
        # Sequential processing
        for combo in tqdm(param_combinations):
            params_dict = dict(zip(param_keys, combo))
            results_batch = run_replications(params_dict, n_replications)
            all_results.extend(results_batch)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/sim_results_final_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(all_results, f)
    
    return all_results


#-------------------------------------------------------
# Visualization Functions
#-------------------------------------------------------

def plot_dilution_effect(dataset: Dict) -> plt.Figure:
    """
    Visualize the effect of dilution on biomarker concentrations.
    """
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    dilution_factors = dataset['dilution_factors']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot true vs observed for first biomarker
    axes[0].scatter(X_true[:, 0], X_obs[:, 0], alpha=0.7, c=dilution_factors, cmap='viridis')
    axes[0].set_xlabel('True Concentration')
    axes[0].set_ylabel('Observed Concentration')
    axes[0].set_title('True vs Observed Concentrations')
    axes[0].plot([0, X_true[:, 0].max()], [0, X_true[:, 0].max()], 'r--')
    
    # Plot dilution factor distribution
    sns.histplot(dilution_factors, kde=True, ax=axes[1])
    axes[1].set_xlabel('Dilution Factor')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Dilution Factor Distribution\n(α={dataset["params"]["dilution_alpha"]}, β={dataset["params"]["dilution_beta"]})')
    
    fig.tight_layout()
    return fig


def plot_normalization_comparison(dataset: Dict) -> plt.Figure:
    """
    Visualize the effect of different normalization methods.
    """
    X_true = dataset['X_true']
    X_obs = dataset['X_obs']
    
    # Apply different normalizations
    methods = ['none', 'total_sum', 'pqn', 'clr']
    normalized_data = {
        'none': X_obs,
        'total_sum': normalize_total_sum(X_obs),
        'pqn': normalize_probabilistic_quotient(X_obs),
        'clr': centered_log_ratio(X_obs)
    }
    
    # Create figure
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8))
    
    # Plot PCA for each method
    for i, method in enumerate(methods):
        X_norm = normalized_data[method]
        
        # Apply PCA
        pca = decomposition.PCA(n_components=2)
        X_pca = pca.fit_transform(X_norm)
        
        # Plot PCA with dilution factor coloring
        scatter = axes[0, i].scatter(X_pca[:, 0], X_pca[:, 1], c=dataset['dilution_factors'], cmap='viridis', alpha=0.7)
        axes[0, i].set_xlabel('PC1')
        axes[0, i].set_ylabel('PC2')
        axes[0, i].set_title(f'{method.capitalize()} - PCA')
        
        # Plot correlation between first two biomarkers
        if method == 'clr':
            corr = np.corrcoef(X_norm[:, 0], X_norm[:, 1])[0, 1]
            axes[1, i].scatter(X_norm[:, 0], X_norm[:, 1], alpha=0.7)
        else:
            corr = np.corrcoef(X_norm[:, 0], X_norm[:, 1])[0, 1]
            axes[1, i].scatter(X_norm[:, 0], X_norm[:, 1], alpha=0.7)
        
        axes[1, i].set_xlabel('Biomarker 1')
        axes[1, i].set_ylabel('Biomarker 2')
        axes[1, i].set_title(f'{method.capitalize()} - Correlation: {corr:.2f}')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    fig.colorbar(scatter, cax=cbar_ax, label='Dilution Factor')
    
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    return fig
