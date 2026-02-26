"""
Comprehensive Unit Tests for Biomarker Dilution Simulation Enhancements

This test suite covers:
1. New normalization methods
2. Enhanced statistical analysis
3. Advanced ML methods
4. New dilution models
5. Visualization functions
6. Data I/O and reproducibility
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from biomarker_dilution_sim import (
    # Data generation
    generate_dataset, generate_correlation_matrix, generate_group_means,
    generate_dilution_factors, generate_dilution_factors_time_dependent,
    generate_dilution_factors_covariate_dependent, simulate_batch_effects,

    # Normalization methods
    normalize_data, normalize_median, normalize_vsn, normalize_ruv,
    normalize_combat, normalize_loess, normalize_total_sum,
    normalize_probabilistic_quotient, centered_log_ratio,

    # Statistical analysis
    multiple_testing_correction, calculate_effect_size, calculate_eta_squared,
    bootstrap_confidence_interval, power_analysis, sample_size_estimation,
    analyze_univariate_enhanced, analyze_univariate,

    # ML methods
    analyze_classification_advanced, feature_selection,
    cross_validate_classification, ensemble_classification,

    # Analysis
    analyze_pca, analyze_clustering, analyze_correlation
)

from visualization_module import (
    plot_volcano, plot_roc_curves_comparison, plot_pca_3d,
    plot_forest, plot_heatmap_clustered, plot_method_comparison_radar,
    plot_dilution_distribution_comparison, plot_power_curve
)

from data_io import (
    SimulationConfig, ExperimentTracker, ReproducibilityManager,
    load_biomarker_data, save_biomarker_data, save_results, load_results,
    results_to_dataframe, compute_data_hash, verify_data_integrity,
    create_experiment
)


class TestNormalizationMethods(unittest.TestCase):
    """Test new normalization methods."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        np.random.seed(42)
        cls.n_samples = 50
        cls.n_biomarkers = 10
        cls.X = np.random.lognormal(0, 1, (cls.n_samples, cls.n_biomarkers))
        cls.y = np.array([0] * 25 + [1] * 25)

    def test_median_normalization(self):
        """Test median normalization."""
        X_norm = normalize_median(self.X)

        self.assertEqual(X_norm.shape, self.X.shape)
        self.assertTrue(np.all(np.isfinite(X_norm)))
        # After median normalization, medians should be more similar
        medians = np.median(X_norm, axis=1)
        self.assertTrue(np.std(medians) < np.std(np.median(self.X, axis=1)))

    def test_vsn_normalization(self):
        """Test VSN normalization."""
        X_norm = normalize_vsn(self.X)

        self.assertEqual(X_norm.shape, self.X.shape)
        self.assertTrue(np.all(np.isfinite(X_norm)))
        # VSN should reduce variance heterogeneity
        self.assertTrue(X_norm.min() < 0)  # asinh produces negative values

    def test_ruv_normalization(self):
        """Test RUV normalization."""
        X_norm = normalize_ruv(self.X, y=self.y, k=1)

        self.assertEqual(X_norm.shape, self.X.shape)
        self.assertTrue(np.all(np.isfinite(X_norm)))
        self.assertTrue(np.all(X_norm >= 0))

    def test_combat_normalization(self):
        """Test ComBat normalization."""
        # Create batch labels
        batch = np.array([0] * 25 + [1] * 25)
        X_norm = normalize_combat(self.X, batch=batch, y=self.y)

        self.assertEqual(X_norm.shape, self.X.shape)
        self.assertTrue(np.all(np.isfinite(X_norm)))

    def test_loess_normalization(self):
        """Test LOESS normalization."""
        X_norm = normalize_loess(self.X, reference_idx=0)

        self.assertEqual(X_norm.shape, self.X.shape)
        self.assertTrue(np.all(np.isfinite(X_norm)))
        self.assertTrue(np.all(X_norm >= 0))

    def test_normalize_data_dispatcher(self):
        """Test normalize_data dispatcher function."""
        methods = ['none', 'total_sum', 'median', 'vsn', 'pqn', 'clr', 'quantile']

        for method in methods:
            X_norm = normalize_data(self.X, method=method)
            self.assertEqual(X_norm.shape, self.X.shape)
            self.assertTrue(np.all(np.isfinite(X_norm)),
                          f"Method {method} produced non-finite values")


class TestStatisticalAnalysis(unittest.TestCase):
    """Test enhanced statistical analysis functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        np.random.seed(42)
        cls.n_samples = 100
        cls.n_biomarkers = 20
        cls.X = np.random.randn(cls.n_samples, cls.n_biomarkers)
        cls.y = np.array([0] * 50 + [1] * 50)

        # Add some true effects
        cls.X[cls.y == 1, :5] += 1.0

    def test_multiple_testing_correction_bonferroni(self):
        """Test Bonferroni correction."""
        p_values = np.array([0.001, 0.01, 0.02, 0.05, 0.1])
        corrected, significant = multiple_testing_correction(
            p_values, method='bonferroni', alpha=0.05
        )

        self.assertEqual(len(corrected), len(p_values))
        self.assertTrue(np.all(corrected >= p_values))
        self.assertTrue(np.all(corrected <= 1.0))

    def test_multiple_testing_correction_fdr(self):
        """Test FDR (Benjamini-Hochberg) correction."""
        p_values = np.array([0.001, 0.01, 0.02, 0.05, 0.1])
        corrected, significant = multiple_testing_correction(
            p_values, method='fdr_bh', alpha=0.05
        )

        self.assertEqual(len(corrected), len(p_values))
        # FDR should be less conservative than Bonferroni
        bonf_corrected, _ = multiple_testing_correction(p_values, 'bonferroni')
        self.assertTrue(np.all(corrected <= bonf_corrected))

    def test_effect_size_cohens_d(self):
        """Test Cohen's d calculation."""
        effect_sizes = calculate_effect_size(self.X, self.y, method='cohens_d')

        self.assertEqual(len(effect_sizes), self.n_biomarkers)
        self.assertTrue(np.all(np.isfinite(effect_sizes)))
        # First 5 biomarkers should have larger effect sizes
        self.assertTrue(np.mean(np.abs(effect_sizes[:5])) >
                       np.mean(np.abs(effect_sizes[5:])))

    def test_effect_size_hedges_g(self):
        """Test Hedges' g calculation."""
        effect_sizes = calculate_effect_size(self.X, self.y, method='hedges_g')

        self.assertEqual(len(effect_sizes), self.n_biomarkers)
        self.assertTrue(np.all(np.isfinite(effect_sizes)))

    def test_eta_squared(self):
        """Test eta-squared calculation."""
        eta_sq = calculate_eta_squared(self.X, self.y)

        self.assertEqual(len(eta_sq), self.n_biomarkers)
        self.assertTrue(np.all(eta_sq >= 0))
        self.assertTrue(np.all(eta_sq <= 1))

    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence intervals."""
        def mean_effect(X, y):
            return np.mean(X[y == 1], axis=0) - np.mean(X[y == 0], axis=0)

        point, lower, upper = bootstrap_confidence_interval(
            self.X, self.y, mean_effect, n_bootstrap=100
        )

        self.assertEqual(len(point), self.n_biomarkers)
        self.assertTrue(np.all(lower <= point))
        self.assertTrue(np.all(point <= upper))

    def test_power_analysis(self):
        """Test power analysis calculation."""
        power = power_analysis(effect_size=0.5, n_per_group=50)

        self.assertGreater(power, 0)
        self.assertLessEqual(power, 1)

        # Larger sample size should give more power
        power_large = power_analysis(effect_size=0.5, n_per_group=100)
        self.assertGreater(power_large, power)

    def test_sample_size_estimation(self):
        """Test sample size estimation."""
        n_required = sample_size_estimation(effect_size=0.5, power=0.8)

        self.assertIsInstance(n_required, int)
        self.assertGreater(n_required, 0)

        # Verify the power at estimated sample size
        achieved_power = power_analysis(0.5, n_required)
        self.assertGreaterEqual(achieved_power, 0.8)

    def test_analyze_univariate_enhanced(self):
        """Test enhanced univariate analysis."""
        results = analyze_univariate_enhanced(self.X, self.y)

        self.assertIn('p_values', results)
        self.assertIn('p_adjusted', results)
        self.assertIn('significant', results)
        self.assertIn('effect_sizes', results)
        self.assertIn('fold_changes', results)


class TestMLMethods(unittest.TestCase):
    """Test advanced ML methods."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        np.random.seed(42)
        cls.n_samples = 100
        cls.n_features = 10
        cls.X = np.random.randn(cls.n_samples, cls.n_features)
        cls.y = np.array([0] * 50 + [1] * 50)

        # Add signal
        cls.X[cls.y == 1, :3] += 1.5

    def test_advanced_classification_gradient_boosting(self):
        """Test gradient boosting classification."""
        X_train, X_test = self.X[:80], self.X[80:]
        y_train, y_test = self.y[:80], self.y[80:]

        y_proba, info = analyze_classification_advanced(
            X_train, y_train, X_test, method='gradient_boosting'
        )

        self.assertEqual(y_proba.shape, (20, 2))
        self.assertTrue(np.allclose(y_proba.sum(axis=1), 1.0))
        self.assertIn('method', info)

    def test_advanced_classification_mlp(self):
        """Test MLP classification."""
        X_train, X_test = self.X[:80], self.X[80:]
        y_train, y_test = self.y[:80], self.y[80:]

        y_proba, info = analyze_classification_advanced(
            X_train, y_train, X_test, method='mlp'
        )

        self.assertEqual(y_proba.shape, (20, 2))

    def test_feature_selection_lasso(self):
        """Test LASSO feature selection."""
        selected, importances = feature_selection(
            self.X, self.y, method='lasso', n_features=5
        )

        self.assertLessEqual(len(selected), 5)
        self.assertGreater(len(selected), 0)
        self.assertEqual(len(importances), self.n_features)

    def test_feature_selection_mutual_info(self):
        """Test mutual information feature selection."""
        selected, importances = feature_selection(
            self.X, self.y, method='mutual_info', n_features=5
        )

        self.assertLessEqual(len(selected), 5)
        self.assertTrue(np.all(importances >= 0))

    def test_feature_selection_random_forest(self):
        """Test random forest feature selection."""
        selected, importances = feature_selection(
            self.X, self.y, method='random_forest', n_features=5
        )

        self.assertLessEqual(len(selected), 5)
        self.assertTrue(np.all(importances >= 0))
        self.assertAlmostEqual(np.sum(importances), 1.0, places=5)

    def test_cross_validation(self):
        """Test cross-validation."""
        results = cross_validate_classification(
            self.X, self.y, method='logistic', n_folds=3
        )

        self.assertIn('overall', results)
        self.assertIn('fold_metrics', results)
        self.assertIn('summary', results)
        self.assertEqual(len(results['fold_metrics']), 3)

    def test_ensemble_classification(self):
        """Test ensemble classification."""
        X_train, X_test = self.X[:80], self.X[80:]
        y_train, y_test = self.y[:80], self.y[80:]

        ensemble_proba, model_results = ensemble_classification(
            X_train, y_train, X_test,
            methods=['logistic', 'random_forest']
        )

        self.assertEqual(ensemble_proba.shape, (20, 2))
        self.assertIn('logistic', model_results)
        self.assertIn('random_forest', model_results)


class TestDilutionModels(unittest.TestCase):
    """Test new dilution models."""

    def test_beta_dilution(self):
        """Test beta distribution dilution."""
        factors = generate_dilution_factors(100, 5.0, 5.0, distribution='beta')

        self.assertEqual(len(factors), 100)
        self.assertTrue(np.all(factors > 0))
        self.assertTrue(np.all(factors < 1))

    def test_gamma_dilution(self):
        """Test gamma distribution dilution."""
        factors = generate_dilution_factors(100, 5.0, 5.0, distribution='gamma')

        self.assertEqual(len(factors), 100)
        self.assertTrue(np.all(factors > 0))
        self.assertTrue(np.all(factors < 1))

    def test_uniform_dilution(self):
        """Test uniform distribution dilution."""
        factors = generate_dilution_factors(100, 5.0, 5.0, distribution='uniform')

        self.assertEqual(len(factors), 100)
        self.assertTrue(np.all(factors >= 0.2))
        self.assertTrue(np.all(factors <= 0.8))

    def test_bimodal_dilution(self):
        """Test bimodal distribution dilution."""
        factors = generate_dilution_factors(1000, 5.0, 5.0, distribution='bimodal')

        self.assertEqual(len(factors), 1000)
        self.assertTrue(np.all(factors > 0))
        self.assertTrue(np.all(factors < 1))

    def test_mixture_dilution(self):
        """Test mixture distribution dilution."""
        factors = generate_dilution_factors(100, 5.0, 5.0, distribution='mixture')

        self.assertEqual(len(factors), 100)
        self.assertTrue(np.all(factors > 0))
        self.assertTrue(np.all(factors < 1))

    def test_time_dependent_dilution(self):
        """Test time-dependent dilution factors."""
        factors = generate_dilution_factors_time_dependent(
            n_subjects=10, n_timepoints=5
        )

        self.assertEqual(factors.shape, (10, 5))
        self.assertTrue(np.all(factors > 0))
        self.assertTrue(np.all(factors < 1))

    def test_covariate_dependent_dilution(self):
        """Test covariate-dependent dilution factors."""
        covariates = {
            'age': np.random.uniform(20, 80, 100),
            'bmi': np.random.uniform(18, 35, 100)
        }
        factors = generate_dilution_factors_covariate_dependent(
            100, covariates
        )

        self.assertEqual(len(factors), 100)
        self.assertTrue(np.all(factors > 0))
        self.assertTrue(np.all(factors < 1))

    def test_batch_effects(self):
        """Test batch effect simulation."""
        batch_assignments, multipliers = simulate_batch_effects(100, 3)

        self.assertEqual(len(batch_assignments), 100)
        self.assertEqual(len(multipliers), 100)
        self.assertTrue(np.all(np.isin(batch_assignments, [0, 1, 2])))


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        np.random.seed(42)
        cls.n_samples = 50
        cls.n_biomarkers = 10
        cls.X = np.random.randn(cls.n_samples, cls.n_biomarkers)
        cls.y = np.array([0] * 25 + [1] * 25)
        cls.fold_changes = np.random.randn(cls.n_biomarkers)
        cls.p_values = np.random.uniform(0, 0.1, cls.n_biomarkers)

    def test_volcano_plot(self):
        """Test volcano plot generation."""
        fig = plot_volcano(self.fold_changes, self.p_values)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_roc_curves(self):
        """Test ROC curve comparison plot."""
        predictions = {
            'method1': np.random.rand(self.n_samples, 2),
            'method2': np.random.rand(self.n_samples, 2)
        }
        # Normalize to sum to 1
        for k in predictions:
            predictions[k] = predictions[k] / predictions[k].sum(axis=1, keepdims=True)

        fig = plot_roc_curves_comparison(self.y, predictions)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_pca_3d(self):
        """Test 3D PCA plot."""
        fig = plot_pca_3d(self.X, self.y)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_forest_plot(self):
        """Test forest plot."""
        effect_sizes = np.random.randn(self.n_biomarkers)
        ci_lower = effect_sizes - 0.5
        ci_upper = effect_sizes + 0.5

        fig = plot_forest(effect_sizes, ci_lower, ci_upper)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_heatmap_clustered(self):
        """Test clustered heatmap."""
        fig = plot_heatmap_clustered(self.X, self.y)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDataIO(unittest.TestCase):
    """Test data I/O and reproducibility features."""

    def test_simulation_config_defaults(self):
        """Test SimulationConfig with defaults."""
        config = SimulationConfig()

        self.assertEqual(config.get('simulation', 'n_subjects'), 100)
        self.assertEqual(config.get('simulation', 'n_biomarkers'), 10)

    def test_simulation_config_custom(self):
        """Test SimulationConfig with custom values."""
        custom = {'simulation': {'n_subjects': 200}}
        config = SimulationConfig(custom)

        self.assertEqual(config.get('simulation', 'n_subjects'), 200)
        self.assertEqual(config.get('simulation', 'n_biomarkers'), 10)

    def test_simulation_config_save_load(self):
        """Test config save and load."""
        config = SimulationConfig()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config.to_file(f.name)
            loaded = SimulationConfig.from_file(f.name)
            os.unlink(f.name)

        self.assertEqual(config.to_dict(), loaded.to_dict())

    def test_reproducibility_manager(self):
        """Test reproducibility manager."""
        repro = ReproducibilityManager(master_seed=42)

        # Generate some random numbers
        arr1 = np.random.rand(10)

        # Reset and generate again
        repro.reset()
        arr2 = np.random.rand(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_data_hash(self):
        """Test data hashing."""
        X = np.random.rand(10, 5)
        hash1 = compute_data_hash(X)
        hash2 = compute_data_hash(X)

        self.assertEqual(hash1, hash2)
        self.assertTrue(verify_data_integrity(X, hash1))

    def test_save_load_biomarker_data(self):
        """Test biomarker data save/load."""
        X = np.random.rand(20, 5)
        y = np.array([0] * 10 + [1] * 10)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            save_biomarker_data(f.name, X, y)
            X_loaded, y_loaded, names, df = load_biomarker_data(
                f.name, group_column='Group'
            )
            os.unlink(f.name)

        self.assertEqual(X_loaded.shape, X.shape)
        np.testing.assert_array_almost_equal(X_loaded, X, decimal=10)

    def test_results_to_dataframe(self):
        """Test results conversion to DataFrame."""
        results = {
            'params': {'n_subjects': 100, 'n_biomarkers': 10},
            'univariate': {
                'none': {'power': 0.8, 'type_i_error': 0.05},
                'pqn': {'power': 0.85, 'type_i_error': 0.04}
            }
        }

        df = results_to_dataframe(results)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df.columns), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_full_simulation_pipeline(self):
        """Test complete simulation pipeline."""
        np.random.seed(42)

        # Generate dataset
        params = {
            'n_subjects': 50,
            'n_biomarkers': 10,
            'n_groups': 2,
            'effect_size': 0.8,
            'distribution': 'lognormal'
        }
        dataset = generate_dataset(**params)

        self.assertIn('X_true', dataset)
        self.assertIn('X_obs', dataset)
        self.assertIn('y', dataset)

        # Apply normalization
        X_norm = normalize_data(dataset['X_obs'], method='pqn')
        self.assertEqual(X_norm.shape, dataset['X_obs'].shape)

        # Statistical analysis
        results = analyze_univariate_enhanced(X_norm, dataset['y'])
        self.assertIn('p_values', results)

        # ML classification
        cv_results = cross_validate_classification(
            X_norm, dataset['y'], method='logistic', n_folds=3
        )
        self.assertIn('overall', cv_results)

    def test_experiment_tracking(self):
        """Test experiment tracking workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker('test_exp', output_dir=tmpdir)

            tracker.log_config({'test': 'config'})
            tracker.log_metric('accuracy', 0.95)
            tracker.log_metrics({'precision': 0.9, 'recall': 0.85})
            tracker.finish('completed')

            # Check files were created
            self.assertTrue((tracker.get_output_dir() / 'metadata.json').exists())
            self.assertTrue((tracker.get_output_dir() / 'metrics.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
