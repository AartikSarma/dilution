import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
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
    beta: float = 5.0,
    distribution: str = 'beta',
    **kwargs
) -> np.ndarray:
    """
    Generate dilution factors from various distributions.

    Parameters:
    -----------
    n_subjects : int
        Number of subjects
    alpha : float
        First shape parameter
    beta : float
        Second shape parameter
    distribution : str
        Distribution type: 'beta', 'gamma', 'uniform', 'truncated_normal',
        'mixture', 'bimodal', 'sample_type'
    **kwargs : dict
        Additional distribution-specific parameters

    Returns:
    --------
    np.ndarray
        Dilution factors (0-1 range for most distributions)
    """
    if distribution == 'beta':
        # Standard Beta distribution
        return np.random.beta(alpha, beta, size=n_subjects)

    elif distribution == 'gamma':
        # Gamma distribution (scaled to 0-1 range)
        shape = kwargs.get('shape', alpha)
        scale = kwargs.get('scale', 1.0 / beta)
        factors = np.random.gamma(shape, scale, size=n_subjects)
        # Normalize to 0-1 range
        factors = factors / (factors.max() + 0.1)
        return np.clip(factors, 0.01, 0.99)

    elif distribution == 'uniform':
        # Uniform distribution
        low = kwargs.get('low', 0.2)
        high = kwargs.get('high', 0.8)
        return np.random.uniform(low, high, size=n_subjects)

    elif distribution == 'truncated_normal':
        # Truncated normal distribution
        mean = kwargs.get('mean', 0.5)
        std = kwargs.get('std', 0.15)
        factors = np.random.normal(mean, std, size=n_subjects)
        return np.clip(factors, 0.01, 0.99)

    elif distribution == 'mixture':
        # Mixture of distributions (simulates different sample quality)
        n_components = kwargs.get('n_components', 2)
        weights = kwargs.get('weights', None)

        if weights is None:
            weights = np.ones(n_components) / n_components

        # Generate component assignments
        components = np.random.choice(n_components, size=n_subjects, p=weights)

        factors = np.zeros(n_subjects)
        for c in range(n_components):
            mask = components == c
            n_c = np.sum(mask)
            if n_c > 0:
                # Each component has different parameters
                alpha_c = alpha * (c + 1) / n_components
                beta_c = beta * (n_components - c) / n_components
                factors[mask] = np.random.beta(
                    max(0.5, alpha_c), max(0.5, beta_c), size=n_c
                )

        return factors

    elif distribution == 'bimodal':
        # Bimodal distribution (good vs poor quality samples)
        p_good = kwargs.get('p_good', 0.6)
        good_params = kwargs.get('good_params', (8.0, 2.0))
        poor_params = kwargs.get('poor_params', (2.0, 8.0))

        is_good = np.random.random(n_subjects) < p_good
        factors = np.zeros(n_subjects)
        factors[is_good] = np.random.beta(good_params[0], good_params[1],
                                          size=np.sum(is_good))
        factors[~is_good] = np.random.beta(poor_params[0], poor_params[1],
                                           size=np.sum(~is_good))
        return factors

    elif distribution == 'sample_type':
        # Different dilution based on sample type
        # (e.g., BAL vs tracheal aspirate vs sputum)
        sample_types = kwargs.get('sample_types', None)
        type_params = kwargs.get('type_params', {
            'BAL': (3.0, 7.0),  # More diluted
            'tracheal': (5.0, 5.0),  # Moderate
            'sputum': (7.0, 3.0)  # Less diluted
        })

        if sample_types is None:
            # Generate random sample types
            types = list(type_params.keys())
            sample_types = np.random.choice(types, size=n_subjects)

        factors = np.zeros(n_subjects)
        for stype, params in type_params.items():
            mask = sample_types == stype
            n_type = np.sum(mask)
            if n_type > 0:
                factors[mask] = np.random.beta(params[0], params[1], size=n_type)

        return factors

    else:
        raise ValueError(f"Unknown dilution distribution: {distribution}")


def generate_dilution_factors_time_dependent(
    n_subjects: int,
    n_timepoints: int,
    base_alpha: float = 5.0,
    base_beta: float = 5.0,
    trend: str = 'none',
    autocorrelation: float = 0.5
) -> np.ndarray:
    """
    Generate time-dependent dilution factors for longitudinal studies.

    Parameters:
    -----------
    n_subjects : int
        Number of subjects
    n_timepoints : int
        Number of time points per subject
    base_alpha, base_beta : float
        Base Beta distribution parameters
    trend : str
        Temporal trend: 'none', 'increasing', 'decreasing', 'cyclic'
    autocorrelation : float
        Autocorrelation between consecutive time points (0-1)

    Returns:
    --------
    np.ndarray
        Dilution factors matrix (n_subjects x n_timepoints)
    """
    factors = np.zeros((n_subjects, n_timepoints))

    for i in range(n_subjects):
        # Generate base subject-level dilution tendency
        subject_tendency = np.random.beta(base_alpha, base_beta)

        for t in range(n_timepoints):
            if t == 0:
                # First time point
                factors[i, t] = np.random.beta(base_alpha, base_beta)
            else:
                # Autocorrelated with previous time point
                prev = factors[i, t - 1]
                noise = np.random.beta(base_alpha, base_beta)
                factors[i, t] = autocorrelation * prev + (1 - autocorrelation) * noise

            # Apply trend
            if trend == 'increasing':
                # Dilution decreases over time (samples get more concentrated)
                trend_factor = 1 + 0.1 * t / n_timepoints
                factors[i, t] = min(0.99, factors[i, t] * trend_factor)

            elif trend == 'decreasing':
                # Dilution increases over time
                trend_factor = 1 - 0.1 * t / n_timepoints
                factors[i, t] = max(0.01, factors[i, t] * trend_factor)

            elif trend == 'cyclic':
                # Cyclic pattern (e.g., circadian)
                cycle_effect = 0.1 * np.sin(2 * np.pi * t / n_timepoints)
                factors[i, t] = np.clip(factors[i, t] + cycle_effect, 0.01, 0.99)

    return factors


def generate_dilution_factors_covariate_dependent(
    n_subjects: int,
    covariates: Dict[str, np.ndarray],
    covariate_effects: Dict[str, float] = None,
    base_alpha: float = 5.0,
    base_beta: float = 5.0
) -> np.ndarray:
    """
    Generate dilution factors that depend on sample/subject covariates.

    Parameters:
    -----------
    n_subjects : int
        Number of subjects
    covariates : dict
        Dictionary of covariate arrays (e.g., {'age': array, 'bmi': array})
    covariate_effects : dict
        Effect sizes for each covariate on dilution
    base_alpha, base_beta : float
        Base Beta distribution parameters

    Returns:
    --------
    np.ndarray
        Dilution factors
    """
    if covariate_effects is None:
        covariate_effects = {k: 0.1 for k in covariates.keys()}

    # Generate base dilution factors
    base_factors = np.random.beta(base_alpha, base_beta, size=n_subjects)

    # Apply covariate effects
    linear_predictor = np.zeros(n_subjects)
    for cov_name, cov_values in covariates.items():
        # Standardize covariate
        cov_std = (cov_values - np.mean(cov_values)) / (np.std(cov_values) + 1e-10)
        effect = covariate_effects.get(cov_name, 0)
        linear_predictor += effect * cov_std

    # Transform to multiplicative effect
    multiplier = np.exp(linear_predictor)
    multiplier = multiplier / np.mean(multiplier)  # Normalize

    # Apply to base factors
    factors = base_factors * multiplier
    factors = np.clip(factors, 0.01, 0.99)

    return factors


def simulate_batch_effects(
    n_subjects: int,
    n_batches: int,
    batch_variability: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate batch effects on dilution factors.

    Parameters:
    -----------
    n_subjects : int
        Number of subjects
    n_batches : int
        Number of processing batches
    batch_variability : float
        Variability between batches (0-1)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (batch assignments, batch-specific dilution multipliers)
    """
    # Assign subjects to batches
    batch_assignments = np.random.randint(0, n_batches, size=n_subjects)

    # Generate batch-specific effects
    batch_effects = 1 + np.random.normal(0, batch_variability, size=n_batches)
    batch_effects = np.maximum(batch_effects, 0.5)  # Ensure positive

    # Get effect for each subject
    dilution_multipliers = batch_effects[batch_assignments]

    return batch_assignments, dilution_multipliers


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

    Parameters:
    -----------
    handling_method : str
        'substitute' : LOD/√2 (Hornung & Reed 1990)
        'zero'       : replace with 0
        'lod_half'   : replace with LOD/2
        'lod'        : replace with LOD value itself
        'mice'       : set below-LOD to NaN, then impute via MICE (IterativeImputer)
        'min'        : replace with minimum observed value
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

        elif handling_method == 'lod_half':
            # Replace with LOD/2
            output[below_lod, j] = lods[j] / 2

        elif handling_method == 'lod':
            # Replace with the LOD value itself
            output[below_lod, j] = lods[j]

        elif handling_method == 'mice':
            # Set below-LOD to NaN; MICE imputation applied after the loop
            output[below_lod, j] = np.nan

        elif handling_method == 'min':
            # Replace with minimum observed value
            if not all(below_lod):
                min_observed = np.min(concentrations[~below_lod, j])
                output[below_lod, j] = min_observed
            else:
                output[below_lod, j] = lods[j] / 2

    # MICE imputation: apply after all columns are marked
    if handling_method == 'mice':
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(
            max_iter=10, random_state=0, min_value=0,
            sample_posterior=False
        )
        output = imputer.fit_transform(output)

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
    biomarker_scales: Optional[np.ndarray] = None,
    ref_group_effect: Optional[float] = None,
    ref_analyte_corr: Optional[Union[float, np.ndarray]] = None,
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
        'lod_percentile': lod_percentile,
        'ref_group_effect': ref_group_effect,
        'ref_analyte_corr': ref_analyte_corr if ref_analyte_corr is None
            else (float(ref_analyte_corr) if np.isscalar(ref_analyte_corr)
                  else ref_analyte_corr.tolist()),
    }
    
    # Generate correlation and covariance matrices
    corr_matrix = generate_correlation_matrix(
        n_biomarkers=n_biomarkers,
        correlation_type=correlation_type,
        block_size=block_size
    )

    # Override reference-analyte correlations if specified
    if ref_analyte_corr is not None:
        rac = np.atleast_1d(ref_analyte_corr)
        if rac.size == 1:
            rac = np.full(n_biomarkers - 1, float(rac))
        corr_matrix[0, 1:] = rac
        corr_matrix[1:, 0] = rac
        # Ensure positive definiteness via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)

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

    # Override reference biomarker (column 0) group effect if specified
    if ref_group_effect is not None:
        for g in range(1, n_groups):
            group_means[g, 0] = group_means[0, 0] * (1 + ref_group_effect)

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
    
    # Determine which biomarkers are truly differential (from known means)
    truly_differential = ~np.isclose(group_means[0], group_means[1])

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
        'truly_differential': truly_differential,
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


def normalize_median(X: np.ndarray) -> np.ndarray:
    """
    Median normalization scales each sample by its median value.

    This is a simple but robust normalization method that divides each
    sample's values by the sample's median, making samples comparable.
    """
    medians = np.median(X, axis=1, keepdims=True)
    # Avoid division by zero
    medians[medians == 0] = np.min(medians[medians > 0])
    return X / medians


def normalize_vsn(X: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Variance Stabilization Normalization (VSN).

    VSN uses a generalized log transformation (asinh) to stabilize variance
    across the intensity range. This implementation uses an iterative approach
    to estimate transformation parameters.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (samples x biomarkers)
    max_iter : int
        Maximum iterations for parameter estimation
    tol : float
        Convergence tolerance

    Returns:
    --------
    np.ndarray
        VSN-transformed data
    """
    n_samples, n_biomarkers = X.shape
    X_positive = np.maximum(X, 1e-10)

    # Initialize parameters
    # Use robust estimates for initialization
    mu = np.median(X_positive, axis=0)
    sigma = np.median(np.abs(X_positive - mu), axis=0) * 1.4826  # MAD to std
    sigma[sigma == 0] = 1.0

    # Generalized log transformation (asinh-based)
    # vsn(x) = asinh((x - mu) / sigma)
    X_vsn = np.zeros_like(X_positive)

    for j in range(n_biomarkers):
        # Iteratively estimate parameters for each biomarker
        x = X_positive[:, j]
        m, s = mu[j], sigma[j]

        for _ in range(max_iter):
            # Transform
            z = (x - m) / s
            transformed = np.arcsinh(z)

            # Update parameters using robust estimates
            m_new = np.median(x - s * np.sinh(transformed))
            residuals = x - m_new
            s_new = np.median(np.abs(residuals)) * 1.4826

            if s_new == 0:
                s_new = s

            # Check convergence
            if abs(m_new - m) < tol and abs(s_new - s) < tol:
                break

            m, s = m_new, s_new

        # Apply final transformation
        X_vsn[:, j] = np.arcsinh((x - m) / s)

    return X_vsn


def normalize_ruv(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    k: int = 1,
    control_genes: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Remove Unwanted Variation (RUV) normalization.

    RUV uses negative control features or factor analysis to identify
    and remove unwanted variation while preserving biological signal.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (samples x biomarkers)
    y : np.ndarray, optional
        Group labels (used to identify unwanted variation)
    k : int
        Number of unwanted factors to remove
    control_genes : np.ndarray, optional
        Boolean mask indicating control biomarkers (not affected by condition)

    Returns:
    --------
    np.ndarray
        RUV-normalized data
    """
    n_samples, n_biomarkers = X.shape

    # Log transform for stability
    X_log = np.log1p(np.maximum(X, 0))

    # If no control genes specified, use residuals from group means
    if control_genes is None:
        if y is not None:
            # Use residuals after removing group effects
            X_residuals = X_log.copy()
            for group in np.unique(y):
                mask = y == group
                group_mean = np.mean(X_log[mask], axis=0)
                X_residuals[mask] -= group_mean
        else:
            # Use all features, centered
            X_residuals = X_log - np.mean(X_log, axis=0)
    else:
        # Use only control genes
        X_residuals = X_log[:, control_genes] - np.mean(X_log[:, control_genes], axis=0)

    # SVD to find unwanted factors
    U, S, Vt = np.linalg.svd(X_residuals, full_matrices=False)

    # Keep only k factors
    k = min(k, len(S))
    W = U[:, :k]  # Unwanted factors

    # Remove unwanted variation from all features
    # X_corrected = X - W @ (W.T @ X)
    alpha = W.T @ X_log  # Coefficients for unwanted factors
    X_corrected = X_log - W @ alpha

    # Transform back from log scale
    X_ruv = np.expm1(X_corrected)
    X_ruv = np.maximum(X_ruv, 0)  # Ensure non-negative

    return X_ruv


def normalize_combat(
    X: np.ndarray,
    batch: np.ndarray,
    y: Optional[np.ndarray] = None,
    parametric: bool = True
) -> np.ndarray:
    """
    ComBat batch correction for removing batch effects.

    This is a simplified implementation of the ComBat algorithm that
    adjusts for batch effects while preserving biological variation.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (samples x biomarkers)
    batch : np.ndarray
        Batch labels for each sample
    y : np.ndarray, optional
        Biological condition labels to preserve
    parametric : bool
        Use parametric (True) or non-parametric (False) adjustment

    Returns:
    --------
    np.ndarray
        Batch-corrected data
    """
    n_samples, n_biomarkers = X.shape
    unique_batches = np.unique(batch)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return X.copy()

    # Log transform for stability
    X_log = np.log1p(np.maximum(X, 0))

    # Step 1: Standardize data
    # Create design matrix for biological covariates
    if y is not None:
        unique_groups = np.unique(y)
        n_groups = len(unique_groups)
        design = np.zeros((n_samples, n_groups))
        for i, group in enumerate(unique_groups):
            design[y == group, i] = 1
        # Fit and get residuals
        coeffs = np.linalg.lstsq(design, X_log, rcond=None)[0]
        X_fitted = design @ coeffs
    else:
        X_fitted = np.mean(X_log, axis=0, keepdims=True).repeat(n_samples, axis=0)

    # Grand mean and pooled variance
    grand_mean = np.mean(X_log, axis=0)
    pooled_var = np.var(X_log, axis=0, ddof=1)
    pooled_var[pooled_var == 0] = 1e-10

    # Standardize
    X_std = (X_log - grand_mean) / np.sqrt(pooled_var)

    # Step 2: Estimate batch effects
    gamma_hat = np.zeros((n_batches, n_biomarkers))  # Location
    delta_hat = np.zeros((n_batches, n_biomarkers))  # Scale

    for i, b in enumerate(unique_batches):
        batch_mask = batch == b
        batch_data = X_std[batch_mask]
        gamma_hat[i] = np.mean(batch_data, axis=0)
        delta_hat[i] = np.var(batch_data, axis=0, ddof=1)

    delta_hat[delta_hat == 0] = 1e-10

    # Step 3: Empirical Bayes shrinkage (parametric)
    if parametric:
        # Estimate hyperparameters
        gamma_bar = np.mean(gamma_hat, axis=0)
        tau2 = np.var(gamma_hat, axis=0, ddof=1)
        tau2[tau2 == 0] = 1e-10

        # Shrink estimates
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        for i in range(n_batches):
            n_batch = np.sum(batch == unique_batches[i])
            # Posterior mean for gamma
            gamma_star[i] = (tau2 * gamma_hat[i] + (1/n_batch) * gamma_bar) / (tau2 + 1/n_batch)
            # Use pooled variance estimate for delta
            delta_star[i] = delta_hat[i]
    else:
        gamma_star = gamma_hat
        delta_star = delta_hat

    # Step 4: Adjust data
    X_combat = np.zeros_like(X_log)

    for i, b in enumerate(unique_batches):
        batch_mask = batch == b
        # Remove batch effect
        adjusted = (X_std[batch_mask] - gamma_star[i]) / np.sqrt(delta_star[i])
        # Add back grand mean and variance
        X_combat[batch_mask] = adjusted * np.sqrt(pooled_var) + grand_mean

    # Transform back
    X_combat = np.expm1(X_combat)
    X_combat = np.maximum(X_combat, 0)

    return X_combat


def normalize_loess(X: np.ndarray, reference_idx: int = 0, span: float = 0.3) -> np.ndarray:
    """
    LOESS (Locally Estimated Scatterplot Smoothing) normalization.

    Normalizes each biomarker against a reference using local regression,
    which can capture non-linear systematic biases.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (samples x biomarkers)
    reference_idx : int
        Index of reference biomarker
    span : float
        Smoothing span for LOESS (0-1)

    Returns:
    --------
    np.ndarray
        LOESS-normalized data
    """
    from scipy.interpolate import UnivariateSpline

    n_samples, n_biomarkers = X.shape
    X_norm = X.copy()

    reference = X[:, reference_idx]
    ref_sorted_idx = np.argsort(reference)
    ref_sorted = reference[ref_sorted_idx]

    for j in range(n_biomarkers):
        if j == reference_idx:
            continue

        target = X[:, j]
        target_sorted = target[ref_sorted_idx]

        # Compute M-A values (log-ratio vs average)
        with np.errstate(divide='ignore', invalid='ignore'):
            M = np.log2(target_sorted + 1) - np.log2(ref_sorted + 1)
            A = 0.5 * (np.log2(target_sorted + 1) + np.log2(ref_sorted + 1))

        # Handle infinities
        valid = np.isfinite(M) & np.isfinite(A)
        if np.sum(valid) < 4:
            continue

        # Fit smoothing spline (approximation of LOESS)
        try:
            # Use smoothing spline as LOESS approximation
            k = min(3, np.sum(valid) - 1)
            if k < 1:
                continue
            spline = UnivariateSpline(A[valid], M[valid], k=k, s=len(A[valid]) * span)
            correction = spline(A)
            correction[~valid] = 0

            # Apply correction
            corrected = np.log2(target_sorted + 1) - correction
            X_norm[ref_sorted_idx, j] = 2**corrected - 1
        except Exception:
            # If spline fitting fails, skip this biomarker
            continue

    X_norm = np.maximum(X_norm, 0)
    return X_norm


def normalize_data(
    X: np.ndarray,
    method: str = 'none',
    reference_idx: Optional[int] = None,
    y: Optional[np.ndarray] = None,
    batch: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Apply normalization method to the data.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix (samples x biomarkers)
    method : str
        Normalization method to apply
    reference_idx : int, optional
        Index of reference biomarker (for reference-based methods)
    y : np.ndarray, optional
        Group labels (for RUV and ComBat)
    batch : np.ndarray, optional
        Batch labels (for ComBat)
    **kwargs : dict
        Additional method-specific parameters

    Returns:
    --------
    np.ndarray
        Normalized data
    """
    # Handle missing or zero values for some methods
    X_clean = X.copy()
    min_positive = np.min(X_clean[X_clean > 0]) if np.any(X_clean > 0) else 1e-10
    X_clean[X_clean <= 0] = min_positive / 10

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
    elif method == 'median':
        return normalize_median(X_clean)
    elif method == 'vsn':
        return normalize_vsn(X_clean, **kwargs)
    elif method == 'ruv':
        k = kwargs.get('k', 1)
        control_genes = kwargs.get('control_genes', None)
        return normalize_ruv(X_clean, y=y, k=k, control_genes=control_genes)
    elif method == 'combat':
        if batch is None:
            # If no batch provided, create synthetic batches based on dilution quartiles
            batch = np.zeros(X.shape[0], dtype=int)
        return normalize_combat(X_clean, batch=batch, y=y, **kwargs)
    elif method == 'loess':
        if reference_idx is None:
            reference_idx = 0
        span = kwargs.get('span', 0.3)
        return normalize_loess(X_clean, reference_idx=reference_idx, span=span)
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


def compute_distance_matrix(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise distance matrix using the specified metric.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (n_samples x n_features)
    metric : str
        Distance metric: 'euclidean', 'bray_curtis', 'aitchison', 'cosine',
        'manhattan', 'canberra', or 'mahalanobis'

    Returns:
    --------
    np.ndarray
        Symmetric pairwise distance matrix (n_samples x n_samples)
    """
    if metric == 'euclidean':
        return squareform(pdist(X, 'euclidean'))
    elif metric == 'bray_curtis':
        # Bray-Curtis requires non-negative data
        X_nn = np.maximum(X, 0)
        return squareform(pdist(X_nn, 'braycurtis'))
    elif metric == 'aitchison':
        # Aitchison distance = Euclidean distance in CLR space
        X_clr = centered_log_ratio(X)
        return squareform(pdist(X_clr, 'euclidean'))
    elif metric == 'cosine':
        X_safe = X.copy()
        row_norms = np.linalg.norm(X_safe, axis=1)
        X_safe[row_norms == 0] += 1e-10
        return squareform(pdist(X_safe, 'cosine'))
    elif metric == 'manhattan':
        return squareform(pdist(X, 'cityblock'))
    elif metric == 'canberra':
        X_nn = np.maximum(X, 0)
        return squareform(pdist(X_nn, 'canberra'))
    elif metric == 'mahalanobis':
        try:
            cov = np.cov(X.T)
            if np.linalg.cond(cov) > 1e10 or X.shape[1] >= X.shape[0]:
                cov += np.eye(cov.shape[0]) * 1e-6
            VI = np.linalg.inv(cov)
            return squareform(pdist(X, 'mahalanobis', VI=VI))
        except (np.linalg.LinAlgError, ValueError):
            variances = np.var(X, axis=0)
            variances[variances == 0] = 1e-10
            VI = np.diag(1.0 / variances)
            return squareform(pdist(X, 'mahalanobis', VI=VI))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


def evaluate_distance_matrix(
    D_true: np.ndarray,
    D_obs: np.ndarray,
    n_permutations: int = 999
) -> Dict:
    """
    Compare two pairwise distance matrices.

    Parameters:
    -----------
    D_true : np.ndarray
        True distance matrix
    D_obs : np.ndarray
        Observed distance matrix
    n_permutations : int
        Number of permutations for Mantel test

    Returns:
    --------
    Dict with mantel_r, mantel_p, rank_correlation, mean_relative_error
    """
    n = D_true.shape[0]
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    d_true_vec = D_true[triu_idx]
    d_obs_vec = D_obs[triu_idx]

    # Mantel correlation (Pearson r between vectorized upper triangles)
    mantel_r, _ = stats.pearsonr(d_true_vec, d_obs_vec)

    # Permutation test for Mantel
    count_ge = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        D_perm = D_obs[np.ix_(perm, perm)]
        d_perm_vec = D_perm[triu_idx]
        r_perm, _ = stats.pearsonr(d_true_vec, d_perm_vec)
        if r_perm >= mantel_r:
            count_ge += 1
    mantel_p = (count_ge + 1) / (n_permutations + 1)

    # Spearman rank correlation
    rank_corr, _ = stats.spearmanr(d_true_vec, d_obs_vec)

    # Mean relative error
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.abs(d_obs_vec - d_true_vec) / np.where(d_true_vec > 0, d_true_vec, np.nan)
    mean_rel_err = float(np.nanmean(rel_err))

    return {
        'mantel_r': float(mantel_r),
        'mantel_p': float(mantel_p),
        'rank_correlation': float(rank_corr),
        'mean_relative_error': mean_rel_err,
    }


def permanova_r2(
    D: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 999
) -> Dict:
    """
    PERMANOVA: partition total sum of squared distances by group labels.

    Parameters:
    -----------
    D : np.ndarray
        Pairwise distance matrix (n x n)
    y : np.ndarray
        Group labels (length n)
    n_permutations : int
        Number of permutations for significance testing

    Returns:
    --------
    Dict with r2, pseudo_f, p_value
    """
    n = len(y)
    groups = np.unique(y)

    # Squared distances
    D2 = D ** 2

    # Total sum of squared distances / n
    ss_total = np.sum(D2) / (2 * n)

    # Within-group sum of squared distances
    ss_within = 0.0
    for g in groups:
        mask = y == g
        n_g = mask.sum()
        if n_g > 1:
            ss_within += np.sum(D2[np.ix_(mask, mask)]) / (2 * n_g)

    ss_between = ss_total - ss_within
    r2 = ss_between / ss_total if ss_total > 0 else 0.0

    # Pseudo-F
    n_groups = len(groups)
    df_between = n_groups - 1
    df_within = n - n_groups
    if df_within > 0 and ss_within > 0:
        pseudo_f = (ss_between / df_between) / (ss_within / df_within)
    else:
        pseudo_f = np.nan

    # Permutation test
    count_ge = 0
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        ss_within_perm = 0.0
        for g in groups:
            mask_p = y_perm == g
            n_g_p = mask_p.sum()
            if n_g_p > 1:
                ss_within_perm += np.sum(D2[np.ix_(mask_p, mask_p)]) / (2 * n_g_p)
        ss_between_perm = ss_total - ss_within_perm
        if df_within > 0 and ss_within_perm > 0:
            f_perm = (ss_between_perm / df_between) / (ss_within_perm / df_within)
        else:
            f_perm = 0.0
        if f_perm >= pseudo_f:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)

    return {
        'r2': float(r2),
        'pseudo_f': float(pseudo_f),
        'p_value': float(p_value),
    }


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
# Advanced Machine Learning Methods
#-------------------------------------------------------

def analyze_classification_advanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str = 'xgboost',
    tune_hyperparameters: bool = False,
    n_cv_folds: int = 5
) -> Tuple[np.ndarray, Dict]:
    """
    Advanced classification with additional methods and hyperparameter tuning.

    Parameters:
    -----------
    X_train : np.ndarray
        Training data
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test data
    method : str
        Classification method: 'xgboost', 'gradient_boosting', 'extra_trees',
        'mlp', 'logistic_l1', 'logistic_l2', 'logistic_elasticnet'
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
    n_cv_folds : int
        Number of cross-validation folds for tuning

    Returns:
    --------
    Tuple[np.ndarray, Dict]
        (predicted probabilities, model info dict)
    """
    from sklearn import linear_model, ensemble, neural_network
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    # Standardize data
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_info = {'method': method, 'tuned': tune_hyperparameters}

    # Define models and parameter grids
    if method == 'xgboost':
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3, 5],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        except ImportError:
            # Fallback to gradient boosting if xgboost not available
            clf = ensemble.GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42
            )
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3, 5],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            model_info['fallback'] = 'gradient_boosting'

    elif method == 'gradient_boosting':
        clf = ensemble.GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 3, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }

    elif method == 'extra_trees':
        clf = ensemble.ExtraTreesClassifier(
            n_estimators=100, random_state=42
        )
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }

    elif method == 'mlp':
        clf = neural_network.MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=500, random_state=42
        )
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }

    elif method == 'logistic_l1':
        clf = linear_model.LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, random_state=42
        )
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0]
        }

    elif method == 'logistic_l2':
        clf = linear_model.LogisticRegression(
            penalty='l2', max_iter=1000, random_state=42
        )
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0]
        }

    elif method == 'logistic_elasticnet':
        clf = linear_model.LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=1000, random_state=42
        )
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }

    else:
        raise ValueError(f"Unknown classification method: {method}")

    # Hyperparameter tuning
    if tune_hyperparameters and len(np.unique(y_train)) > 1:
        try:
            grid_search = GridSearchCV(
                clf, param_grid, cv=n_cv_folds, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            clf = grid_search.best_estimator_
            model_info['best_params'] = grid_search.best_params_
            model_info['cv_score'] = grid_search.best_score_
        except Exception as e:
            # If tuning fails, use default parameters
            clf.fit(X_train_scaled, y_train)
            model_info['tuning_error'] = str(e)
    else:
        clf.fit(X_train_scaled, y_train)

    # Predict probabilities
    y_proba = clf.predict_proba(X_test_scaled)

    # Extract feature importances if available
    if hasattr(clf, 'feature_importances_'):
        model_info['feature_importances'] = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        model_info['coefficients'] = clf.coef_

    return y_proba, model_info


def feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'lasso',
    n_features: Optional[int] = None,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform feature selection using various methods.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (samples x biomarkers)
    y : np.ndarray
        Labels
    method : str
        Selection method: 'lasso', 'rfe', 'mutual_info', 'f_classif',
        'random_forest', 'boruta'
    n_features : int, optional
        Number of features to select
    threshold : float, optional
        Threshold for feature importance

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (selected feature indices, feature scores/importances)
    """
    from sklearn.feature_selection import (
        SelectKBest, f_classif, mutual_info_classif, RFE
    )
    from sklearn import linear_model, ensemble

    n_biomarkers = X.shape[1]

    # Default to selecting half of features
    if n_features is None:
        n_features = max(1, n_biomarkers // 2)

    # Standardize for some methods
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == 'lasso':
        # LASSO-based selection
        lasso = linear_model.LogisticRegression(
            penalty='l1', solver='saga', max_iter=1000, random_state=42
        )
        lasso.fit(X_scaled, y)

        # Get absolute coefficients
        if len(lasso.coef_.shape) > 1:
            importances = np.mean(np.abs(lasso.coef_), axis=0)
        else:
            importances = np.abs(lasso.coef_)

        # Select top features
        if threshold is not None:
            selected = np.where(importances > threshold)[0]
        else:
            selected = np.argsort(importances)[-n_features:]

    elif method == 'rfe':
        # Recursive Feature Elimination
        estimator = linear_model.LogisticRegression(max_iter=1000, random_state=42)
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X_scaled, y)

        selected = np.where(rfe.support_)[0]
        importances = rfe.ranking_

    elif method == 'mutual_info':
        # Mutual information
        importances = mutual_info_classif(X, y, random_state=42)
        selected = np.argsort(importances)[-n_features:]

    elif method == 'f_classif':
        # F-test (ANOVA F-value)
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X, y)

        importances = selector.scores_
        selected = np.where(selector.get_support())[0]

    elif method == 'random_forest':
        # Random Forest importance
        rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = rf.feature_importances_

        if threshold is not None:
            selected = np.where(importances > threshold)[0]
        else:
            selected = np.argsort(importances)[-n_features:]

    elif method == 'boruta':
        # Boruta-like all-relevant feature selection
        # Simplified implementation using shadow features
        n_iterations = 100
        hit_counts = np.zeros(n_biomarkers)

        for _ in range(n_iterations):
            # Create shadow features (shuffled copies)
            X_shadow = X.copy()
            for j in range(n_biomarkers):
                np.random.shuffle(X_shadow[:, j])

            X_combined = np.hstack([X, X_shadow])

            # Train random forest
            rf = ensemble.RandomForestClassifier(n_estimators=50, random_state=None)
            rf.fit(X_combined, y)

            # Get importances
            real_imp = rf.feature_importances_[:n_biomarkers]
            shadow_imp = rf.feature_importances_[n_biomarkers:]
            shadow_max = np.max(shadow_imp)

            # Count hits (real > max shadow)
            hit_counts += (real_imp > shadow_max).astype(int)

        # Select features that beat shadow features consistently
        importances = hit_counts / n_iterations

        if threshold is not None:
            selected = np.where(importances > threshold)[0]
        else:
            selected = np.argsort(importances)[-n_features:]

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    # Ensure we return at least one feature
    if len(selected) == 0:
        selected = np.array([np.argmax(importances)])

    return selected, importances


def cross_validate_classification(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'logistic',
    n_folds: int = 5,
    stratified: bool = True,
    return_predictions: bool = False
) -> Dict:
    """
    Perform cross-validated classification with comprehensive metrics.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    y : np.ndarray
        Labels
    method : str
        Classification method
    n_folds : int
        Number of CV folds
    stratified : bool
        Use stratified folds
    return_predictions : bool
        Return out-of-fold predictions

    Returns:
    --------
    Dict containing CV results
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize storage
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Get predictions
        if method in ['xgboost', 'gradient_boosting', 'extra_trees', 'mlp',
                      'logistic_l1', 'logistic_l2', 'logistic_elasticnet']:
            y_proba, _ = analyze_classification_advanced(
                X_train, y_train, X_test, method
            )
        else:
            y_proba = analyze_classification(X_train, y_train, X_test, method)

        y_pred = np.argmax(y_proba, axis=1)

        # Store predictions
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

        # Calculate fold metrics
        fold_result = evaluate_classification(y_test, y_proba)
        fold_result['fold'] = fold
        fold_metrics.append(fold_result)

    # Aggregate results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    # Calculate overall metrics
    overall_metrics = evaluate_classification(all_y_true, all_y_proba)

    # Calculate mean and std of fold metrics
    metric_names = [k for k in fold_metrics[0].keys() if k != 'fold']
    summary = {}
    for metric in metric_names:
        values = [f[metric] for f in fold_metrics if not np.isnan(f[metric])]
        if values:
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)

    results = {
        'overall': overall_metrics,
        'fold_metrics': fold_metrics,
        'summary': summary,
        'n_folds': n_folds,
        'method': method
    }

    if return_predictions:
        results['y_true'] = all_y_true
        results['y_pred'] = all_y_pred
        results['y_proba'] = all_y_proba

    return results


def ensemble_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    methods: List[str] = None,
    voting: str = 'soft'
) -> Tuple[np.ndarray, Dict]:
    """
    Ensemble classification combining multiple methods.

    Parameters:
    -----------
    X_train, y_train : training data
    X_test : test data
    methods : list of classification methods to ensemble
    voting : 'soft' (probability average) or 'hard' (majority vote)

    Returns:
    --------
    Tuple[np.ndarray, Dict]
        (ensemble predictions, individual model results)
    """
    if methods is None:
        methods = ['logistic', 'random_forest', 'gradient_boosting']

    # Collect predictions from each method
    all_proba = []
    all_pred = []
    model_results = {}

    for method in methods:
        try:
            if method in ['xgboost', 'gradient_boosting', 'extra_trees', 'mlp',
                          'logistic_l1', 'logistic_l2', 'logistic_elasticnet']:
                y_proba, info = analyze_classification_advanced(
                    X_train, y_train, X_test, method
                )
            else:
                y_proba = analyze_classification(X_train, y_train, X_test, method)
                info = {}

            all_proba.append(y_proba)
            all_pred.append(np.argmax(y_proba, axis=1))
            model_results[method] = {'proba': y_proba, 'info': info}
        except Exception as e:
            model_results[method] = {'error': str(e)}

    if len(all_proba) == 0:
        raise ValueError("All classification methods failed")

    # Ensemble predictions
    if voting == 'soft':
        # Average probabilities
        ensemble_proba = np.mean(all_proba, axis=0)
    else:
        # Hard voting (majority)
        all_pred = np.array(all_pred)
        n_classes = all_proba[0].shape[1]
        ensemble_proba = np.zeros((X_test.shape[0], n_classes))
        for i in range(X_test.shape[0]):
            votes = all_pred[:, i]
            for c in range(n_classes):
                ensemble_proba[i, c] = np.sum(votes == c) / len(votes)

    return ensemble_proba, model_results


#-------------------------------------------------------
# Enhanced Statistical Analysis
#-------------------------------------------------------

def multiple_testing_correction(
    p_values: np.ndarray,
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple testing correction to p-values.

    Parameters:
    -----------
    p_values : np.ndarray
        Array of p-values
    method : str
        Correction method: 'bonferroni', 'fdr_bh' (Benjamini-Hochberg),
        'fdr_by' (Benjamini-Yekutieli), 'holm', 'hochberg'
    alpha : float
        Significance level

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (corrected p-values, boolean array of significant results)
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    if method == 'bonferroni':
        # Bonferroni correction
        corrected = np.minimum(p_values * n, 1.0)
        significant = corrected < alpha

    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR
        corrected = np.zeros(n)
        ranks = np.arange(1, n + 1)

        # Calculate adjusted p-values
        adjusted = sorted_p * n / ranks
        # Ensure monotonicity
        cummin = np.minimum.accumulate(adjusted[::-1])[::-1]
        corrected[sorted_idx] = np.minimum(cummin, 1.0)
        significant = corrected < alpha

    elif method == 'fdr_by':
        # Benjamini-Yekutieli FDR (more conservative)
        corrected = np.zeros(n)
        ranks = np.arange(1, n + 1)
        c_n = np.sum(1.0 / ranks)  # Correction factor

        adjusted = sorted_p * n * c_n / ranks
        cummin = np.minimum.accumulate(adjusted[::-1])[::-1]
        corrected[sorted_idx] = np.minimum(cummin, 1.0)
        significant = corrected < alpha

    elif method == 'holm':
        # Holm-Bonferroni step-down
        corrected = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min((n - i) * sorted_p[i], 1.0)
        # Ensure monotonicity
        cummax = np.maximum.accumulate(corrected[sorted_idx])
        corrected[sorted_idx] = cummax
        significant = corrected < alpha

    elif method == 'hochberg':
        # Hochberg step-up
        corrected = np.zeros(n)
        for i in range(n - 1, -1, -1):
            idx = sorted_idx[i]
            corrected[idx] = min((n - i) * sorted_p[i], 1.0)
        # Ensure monotonicity (reverse)
        for i in range(1, n):
            idx = sorted_idx[i]
            prev_idx = sorted_idx[i - 1]
            corrected[idx] = min(corrected[idx], corrected[prev_idx])
        significant = corrected < alpha

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected, significant


def calculate_effect_size(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'cohens_d'
) -> np.ndarray:
    """
    Calculate effect sizes for each biomarker between groups.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (samples x biomarkers)
    y : np.ndarray
        Group labels
    method : str
        Effect size method: 'cohens_d', 'hedges_g', 'glass_delta', 'cliff_delta'

    Returns:
    --------
    np.ndarray
        Effect sizes for each biomarker
    """
    unique_groups = np.unique(y)
    n_biomarkers = X.shape[1]

    if len(unique_groups) != 2:
        # For multi-group, calculate eta-squared instead
        return calculate_eta_squared(X, y)

    group0 = X[y == unique_groups[0]]
    group1 = X[y == unique_groups[1]]

    n0, n1 = len(group0), len(group1)
    mean0 = np.mean(group0, axis=0)
    mean1 = np.mean(group1, axis=0)
    var0 = np.var(group0, axis=0, ddof=1)
    var1 = np.var(group1, axis=0, ddof=1)

    effect_sizes = np.zeros(n_biomarkers)

    if method == 'cohens_d':
        # Cohen's d: standardized mean difference
        pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
        pooled_std[pooled_std == 0] = 1e-10
        effect_sizes = (mean1 - mean0) / pooled_std

    elif method == 'hedges_g':
        # Hedges' g: bias-corrected Cohen's d
        pooled_std = np.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2))
        pooled_std[pooled_std == 0] = 1e-10
        d = (mean1 - mean0) / pooled_std
        # Correction factor for small samples
        correction = 1 - 3 / (4 * (n0 + n1) - 9)
        effect_sizes = d * correction

    elif method == 'glass_delta':
        # Glass's delta: uses control group std only
        std0 = np.sqrt(var0)
        std0[std0 == 0] = 1e-10
        effect_sizes = (mean1 - mean0) / std0

    elif method == 'cliff_delta':
        # Cliff's delta: non-parametric effect size
        for j in range(n_biomarkers):
            x0 = group0[:, j]
            x1 = group1[:, j]
            # Count dominance
            n_greater = np.sum(x1[:, np.newaxis] > x0)
            n_less = np.sum(x1[:, np.newaxis] < x0)
            effect_sizes[j] = (n_greater - n_less) / (n0 * n1)

    return effect_sizes


def calculate_eta_squared(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate eta-squared (effect size for ANOVA) for each biomarker.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (samples x biomarkers)
    y : np.ndarray
        Group labels

    Returns:
    --------
    np.ndarray
        Eta-squared values for each biomarker
    """
    n_biomarkers = X.shape[1]
    unique_groups = np.unique(y)
    eta_squared = np.zeros(n_biomarkers)

    for j in range(n_biomarkers):
        # Total sum of squares
        grand_mean = np.mean(X[:, j])
        ss_total = np.sum((X[:, j] - grand_mean) ** 2)

        # Between-group sum of squares
        ss_between = 0
        for group in unique_groups:
            group_data = X[y == group, j]
            group_mean = np.mean(group_data)
            ss_between += len(group_data) * (group_mean - grand_mean) ** 2

        if ss_total > 0:
            eta_squared[j] = ss_between / ss_total
        else:
            eta_squared[j] = 0

    return eta_squared


def bootstrap_confidence_interval(
    X: np.ndarray,
    y: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate bootstrap confidence intervals for a statistic.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (samples x biomarkers)
    y : np.ndarray
        Group labels
    statistic_func : callable
        Function that takes (X, y) and returns statistic(s)
    n_bootstrap : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level (0-1)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (point estimate, lower CI, upper CI)
    """
    np.random.seed(random_state)
    n_samples = len(y)

    # Calculate point estimate
    point_estimate = statistic_func(X, y)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Calculate statistic
        stat = statistic_func(X_boot, y_boot)
        bootstrap_stats.append(stat)

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate percentile confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_ci = np.percentile(bootstrap_stats, lower_percentile, axis=0)
    upper_ci = np.percentile(bootstrap_stats, upper_percentile, axis=0)

    return point_estimate, lower_ci, upper_ci


def power_analysis(
    effect_size: float,
    n_per_group: int,
    alpha: float = 0.05,
    test_type: str = 't_test'
) -> float:
    """
    Calculate statistical power for a given effect size and sample size.

    Parameters:
    -----------
    effect_size : float
        Expected effect size (Cohen's d for t-test)
    n_per_group : int
        Number of samples per group
    alpha : float
        Significance level
    test_type : str
        Type of test: 't_test' or 'anova'

    Returns:
    --------
    float
        Statistical power (0-1)
    """
    from scipy.stats import norm, ncf

    if test_type == 't_test':
        # Two-sample t-test power calculation
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n_per_group / 2)

        # Degrees of freedom
        df = 2 * n_per_group - 2

        # Critical t-value
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        # Power = P(|T| > t_crit | H1)
        # Using non-central t distribution
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    elif test_type == 'anova':
        # One-way ANOVA power (for 2 groups, equivalent to t-test)
        # Effect size is eta-squared or f
        f_effect = effect_size / np.sqrt(1 - effect_size) if effect_size < 1 else effect_size

        df1 = 1  # Between groups df (for 2 groups)
        df2 = 2 * n_per_group - 2  # Within groups df

        # Non-centrality parameter
        ncp = f_effect ** 2 * 2 * n_per_group

        # Critical F-value
        f_crit = stats.f.ppf(1 - alpha, df1, df2)

        # Power using non-central F distribution
        power = 1 - ncf.cdf(f_crit, df1, df2, ncp)

    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return power


def sample_size_estimation(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = 't_test'
) -> int:
    """
    Estimate required sample size per group for desired power.

    Parameters:
    -----------
    effect_size : float
        Expected effect size (Cohen's d)
    power : float
        Desired statistical power
    alpha : float
        Significance level
    test_type : str
        Type of test

    Returns:
    --------
    int
        Required sample size per group
    """
    # Binary search for sample size
    n_min, n_max = 2, 10000

    while n_max - n_min > 1:
        n_mid = (n_min + n_max) // 2
        achieved_power = power_analysis(effect_size, n_mid, alpha, test_type)

        if achieved_power >= power:
            n_max = n_mid
        else:
            n_min = n_mid

    return n_max


def analyze_univariate_enhanced(
    X: np.ndarray,
    y: np.ndarray,
    test_type: str = 't_test',
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    calculate_effect_sizes: bool = True
) -> Dict:
    """
    Enhanced univariate analysis with multiple testing correction and effect sizes.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix (samples x biomarkers)
    y : np.ndarray
        Group labels
    test_type : str
        Statistical test type
    correction_method : str
        Multiple testing correction method
    alpha : float
        Significance level
    calculate_effect_sizes : bool
        Whether to calculate effect sizes

    Returns:
    --------
    Dict containing:
        - p_values: raw p-values
        - p_adjusted: corrected p-values
        - significant: boolean mask of significant biomarkers
        - test_statistics: test statistics
        - effect_sizes: effect sizes (if calculated)
        - fold_changes: log2 fold changes
    """
    # Perform basic univariate tests
    p_values, test_stats = analyze_univariate(X, y, test_type)

    # Multiple testing correction
    p_adjusted, significant = multiple_testing_correction(p_values, correction_method, alpha)

    results = {
        'p_values': p_values,
        'p_adjusted': p_adjusted,
        'significant': significant,
        'test_statistics': test_stats,
        'n_significant': np.sum(significant)
    }

    # Calculate effect sizes
    if calculate_effect_sizes:
        effect_sizes = calculate_effect_size(X, y, method='cohens_d')
        results['effect_sizes'] = effect_sizes

    # Calculate fold changes (for two groups)
    unique_groups = np.unique(y)
    if len(unique_groups) == 2:
        mean0 = np.mean(X[y == unique_groups[0]], axis=0)
        mean1 = np.mean(X[y == unique_groups[1]], axis=0)
        # Log2 fold change
        with np.errstate(divide='ignore', invalid='ignore'):
            fc = np.log2((mean1 + 1e-10) / (mean0 + 1e-10))
        results['fold_changes'] = fc

    return results

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
