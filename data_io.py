"""
Data I/O and Reproducibility Module for Biomarker Dilution Simulation

This module provides:
- Configuration file support (YAML/JSON)
- Data import/export (CSV, Excel)
- Experiment tracking and logging
- Seed management for reproducibility
"""

import numpy as np
import pandas as pd
import json
import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import warnings

# Try to import optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


#-------------------------------------------------------
# Configuration Management
#-------------------------------------------------------

class SimulationConfig:
    """
    Configuration management for simulation experiments.

    Supports loading/saving configurations from YAML or JSON files,
    with validation and default values.
    """

    DEFAULT_CONFIG = {
        'simulation': {
            'n_subjects': 100,
            'n_biomarkers': 10,
            'n_groups': 2,
            'n_replications': 100,
            'random_seed': 42
        },
        'data_generation': {
            'correlation_type': 'moderate',
            'effect_size': 0.5,
            'distribution': 'lognormal',
            'block_size': None
        },
        'dilution': {
            'distribution': 'beta',
            'alpha': 5.0,
            'beta': 5.0
        },
        'detection_limits': {
            'lod_percentile': 0.1,
            'handling_method': 'substitute'
        },
        'normalization': {
            'methods': ['none', 'total_sum', 'pqn', 'clr']
        },
        'analysis': {
            'test_type': 't_test',
            'correction_method': 'fdr_bh',
            'alpha': 0.05
        },
        'output': {
            'directory': 'results',
            'save_intermediate': True,
            'save_plots': True,
            'format': 'csv'
        }
    }

    def __init__(self, config_dict: Dict = None):
        """
        Initialize configuration with defaults or provided dictionary.

        Parameters:
        -----------
        config_dict : dict, optional
            Configuration dictionary to override defaults
        """
        self.config = self._deep_copy(self.DEFAULT_CONFIG)

        if config_dict is not None:
            self._update_nested(self.config, config_dict)

        self._validate()

    def _deep_copy(self, d: Dict) -> Dict:
        """Create a deep copy of a dictionary."""
        if isinstance(d, dict):
            return {k: self._deep_copy(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._deep_copy(v) for v in d]
        else:
            return d

    def _update_nested(self, base: Dict, update: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_nested(base[key], value)
            else:
                base[key] = value

    def _validate(self) -> None:
        """Validate configuration values."""
        sim = self.config['simulation']

        if sim['n_subjects'] < 2:
            raise ValueError("n_subjects must be at least 2")
        if sim['n_biomarkers'] < 1:
            raise ValueError("n_biomarkers must be at least 1")
        if sim['n_groups'] < 1:
            raise ValueError("n_groups must be at least 1")

        dil = self.config['dilution']
        if dil['alpha'] <= 0 or dil['beta'] <= 0:
            raise ValueError("Dilution alpha and beta must be positive")

        analysis = self.config['analysis']
        if not 0 < analysis['alpha'] < 1:
            raise ValueError("Alpha must be between 0 and 1")

    @classmethod
    def from_file(cls, filepath: str) -> 'SimulationConfig':
        """
        Load configuration from a file (YAML or JSON).

        Parameters:
        -----------
        filepath : str
            Path to configuration file

        Returns:
        --------
        SimulationConfig
            Loaded configuration object
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            if filepath.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to load YAML files. Install with: pip install pyyaml")
                config_dict = yaml.safe_load(f)
            elif filepath.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")

        return cls(config_dict)

    def to_file(self, filepath: str) -> None:
        """
        Save configuration to a file (YAML or JSON).

        Parameters:
        -----------
        filepath : str
            Path to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            if filepath.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to save YAML files")
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2)

    def get(self, *keys, default=None):
        """
        Get a nested configuration value.

        Parameters:
        -----------
        *keys : str
            Nested keys to access
        default : any
            Default value if key not found

        Returns:
        --------
        Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys, value) -> None:
        """
        Set a nested configuration value.

        Parameters:
        -----------
        *keys : str
            Nested keys to access
        value : any
            Value to set
        """
        d = self.config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def to_dict(self) -> Dict:
        """Return configuration as dictionary."""
        return self._deep_copy(self.config)

    def get_simulation_params(self) -> Dict:
        """Get parameters for generate_dataset function."""
        return {
            'n_subjects': self.get('simulation', 'n_subjects'),
            'n_biomarkers': self.get('simulation', 'n_biomarkers'),
            'n_groups': self.get('simulation', 'n_groups'),
            'correlation_type': self.get('data_generation', 'correlation_type'),
            'effect_size': self.get('data_generation', 'effect_size'),
            'distribution': self.get('data_generation', 'distribution'),
            'dilution_alpha': self.get('dilution', 'alpha'),
            'dilution_beta': self.get('dilution', 'beta'),
            'lod_percentile': self.get('detection_limits', 'lod_percentile'),
            'lod_handling': self.get('detection_limits', 'handling_method'),
            'block_size': self.get('data_generation', 'block_size')
        }


#-------------------------------------------------------
# Data Import/Export
#-------------------------------------------------------

def load_biomarker_data(
    filepath: str,
    data_columns: List[str] = None,
    group_column: str = None,
    sample_id_column: str = None,
    sheet_name: str = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]], pd.DataFrame]:
    """
    Load biomarker data from CSV or Excel file.

    Parameters:
    -----------
    filepath : str
        Path to data file
    data_columns : list, optional
        Names of columns containing biomarker data (if None, uses all numeric columns)
    group_column : str, optional
        Name of column containing group labels
    sample_id_column : str, optional
        Name of column containing sample IDs
    sheet_name : str, optional
        Sheet name for Excel files

    Returns:
    --------
    Tuple of:
        - X: np.ndarray of biomarker data
        - y: np.ndarray of group labels (or None)
        - biomarker_names: list of biomarker names
        - df: original DataFrame
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load data
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix in ['.xlsx', '.xls']:
        if not EXCEL_AVAILABLE:
            raise ImportError("openpyxl is required to read Excel files. Install with: pip install openpyxl")
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Determine data columns
    if data_columns is None:
        # Use all numeric columns except group and sample ID columns
        exclude_cols = []
        if group_column:
            exclude_cols.append(group_column)
        if sample_id_column:
            exclude_cols.append(sample_id_column)

        data_columns = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

    # Extract data matrix
    X = df[data_columns].values.astype(float)
    biomarker_names = list(data_columns)

    # Extract group labels if specified
    y = None
    if group_column and group_column in df.columns:
        # Convert to numeric labels
        unique_groups = df[group_column].unique()
        group_map = {g: i for i, g in enumerate(unique_groups)}
        y = df[group_column].map(group_map).values

    return X, y, biomarker_names, df


def save_biomarker_data(
    filepath: str,
    X: np.ndarray,
    y: np.ndarray = None,
    biomarker_names: List[str] = None,
    sample_ids: List[str] = None,
    additional_columns: Dict[str, np.ndarray] = None,
    sheet_name: str = 'data'
) -> None:
    """
    Save biomarker data to CSV or Excel file.

    Parameters:
    -----------
    filepath : str
        Path to save data
    X : np.ndarray
        Biomarker data matrix
    y : np.ndarray, optional
        Group labels
    biomarker_names : list, optional
        Names of biomarkers
    sample_ids : list, optional
        Sample identifiers
    additional_columns : dict, optional
        Additional columns to include {name: array}
    sheet_name : str
        Sheet name for Excel files
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_samples, n_biomarkers = X.shape

    # Create column names
    if biomarker_names is None:
        biomarker_names = [f'Biomarker_{i+1}' for i in range(n_biomarkers)]

    # Create DataFrame
    df = pd.DataFrame(X, columns=biomarker_names)

    # Add sample IDs
    if sample_ids is not None:
        df.insert(0, 'Sample_ID', sample_ids)
    else:
        df.insert(0, 'Sample_ID', [f'S{i+1}' for i in range(n_samples)])

    # Add group labels
    if y is not None:
        df.insert(1, 'Group', y)

    # Add additional columns
    if additional_columns:
        for name, values in additional_columns.items():
            df[name] = values

    # Save file
    if filepath.suffix == '.csv':
        df.to_csv(filepath, index=False)
    elif filepath.suffix in ['.xlsx', '.xls']:
        if not EXCEL_AVAILABLE:
            raise ImportError("openpyxl is required to write Excel files")
        df.to_excel(filepath, sheet_name=sheet_name, index=False)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_results(
    filepath: str,
    results: Dict,
    format: str = 'json'
) -> None:
    """
    Save simulation results to file.

    Parameters:
    -----------
    filepath : str
        Path to save results
    results : dict
        Results dictionary
    format : str
        Output format: 'json', 'csv', or 'pickle'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj

    if format == 'json':
        serializable = convert_to_serializable(results)
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

    elif format == 'pickle':
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

    elif format == 'csv':
        # Flatten results to DataFrame
        df = results_to_dataframe(results)
        df.to_csv(filepath, index=False)

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(filepath: str) -> Dict:
    """
    Load simulation results from file.

    Parameters:
    -----------
    filepath : str
        Path to results file

    Returns:
    --------
    dict
        Results dictionary
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)

    elif filepath.suffix in ['.pkl', '.pickle']:
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def results_to_dataframe(results: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """
    Convert simulation results to a flat DataFrame.

    Parameters:
    -----------
    results : dict or list
        Results from simulation

    Returns:
    --------
    pd.DataFrame
        Flattened results
    """
    if isinstance(results, dict):
        results = [results]

    rows = []
    for res in results:
        row = {}

        # Extract parameters
        if 'params' in res:
            for k, v in res['params'].items():
                row[f'param_{k}'] = v

        # Extract metrics for each normalization method
        for category in ['univariate', 'correlation', 'pca', 'clustering', 'classification']:
            if category in res:
                for method, metrics in res[category].items():
                    if isinstance(metrics, dict):
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float, np.number)):
                                row[f'{category}_{method}_{metric_name}'] = metric_value

        rows.append(row)

    return pd.DataFrame(rows)


#-------------------------------------------------------
# Experiment Tracking
#-------------------------------------------------------

class ExperimentTracker:
    """
    Track and log simulation experiments for reproducibility.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = 'experiments',
        log_level: int = logging.INFO
    ):
        """
        Initialize experiment tracker.

        Parameters:
        -----------
        experiment_name : str
            Name of the experiment
        output_dir : str
            Directory to store experiment outputs
        log_level : int
            Logging level
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"

        # Create output directory
        self.output_dir = Path(output_dir) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging(log_level)

        # Initialize metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'status': 'initialized'
        }

        # Track metrics
        self.metrics = {}
        self.artifacts = []

        self.logger.info(f"Experiment initialized: {self.experiment_id}")

    def _setup_logging(self, log_level: int) -> None:
        """Setup logging for the experiment."""
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(log_level)

        # File handler
        log_file = self.output_dir / 'experiment.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_config(self, config: Union[Dict, SimulationConfig]) -> None:
        """
        Log experiment configuration.

        Parameters:
        -----------
        config : dict or SimulationConfig
            Experiment configuration
        """
        if isinstance(config, SimulationConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config

        self.metadata['config'] = config_dict

        # Save config file
        config_file = self.output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        self.logger.info("Configuration logged")

    def log_metric(self, name: str, value: float, step: int = None) -> None:
        """
        Log a metric value.

        Parameters:
        -----------
        name : str
            Metric name
        value : float
            Metric value
        step : int, optional
            Step or iteration number
        """
        if name not in self.metrics:
            self.metrics[name] = []

        entry = {'value': value, 'timestamp': datetime.now().isoformat()}
        if step is not None:
            entry['step'] = step

        self.metrics[name].append(entry)
        self.logger.debug(f"Metric logged: {name}={value}")

    def log_metrics(self, metrics_dict: Dict[str, float], step: int = None) -> None:
        """
        Log multiple metrics at once.

        Parameters:
        -----------
        metrics_dict : dict
            Dictionary of metric names and values
        step : int, optional
            Step or iteration number
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)

    def save_artifact(
        self,
        name: str,
        data: Any,
        artifact_type: str = 'auto'
    ) -> str:
        """
        Save an artifact (data, figure, etc.).

        Parameters:
        -----------
        name : str
            Artifact name
        data : any
            Data to save
        artifact_type : str
            Type of artifact: 'figure', 'data', 'model', or 'auto'

        Returns:
        --------
        str
            Path to saved artifact
        """
        artifacts_dir = self.output_dir / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)

        # Determine type and save
        if artifact_type == 'auto':
            if hasattr(data, 'savefig'):
                artifact_type = 'figure'
            elif isinstance(data, (np.ndarray, pd.DataFrame)):
                artifact_type = 'data'
            else:
                artifact_type = 'pickle'

        if artifact_type == 'figure':
            filepath = artifacts_dir / f"{name}.png"
            data.savefig(filepath, dpi=150, bbox_inches='tight')

        elif artifact_type == 'data':
            if isinstance(data, pd.DataFrame):
                filepath = artifacts_dir / f"{name}.csv"
                data.to_csv(filepath, index=False)
            else:
                filepath = artifacts_dir / f"{name}.npy"
                np.save(filepath, data)

        else:
            import pickle
            filepath = artifacts_dir / f"{name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

        self.artifacts.append({
            'name': name,
            'type': artifact_type,
            'path': str(filepath)
        })

        self.logger.info(f"Artifact saved: {name} -> {filepath}")
        return str(filepath)

    def finish(self, status: str = 'completed') -> None:
        """
        Finish experiment and save final metadata.

        Parameters:
        -----------
        status : str
            Final status: 'completed', 'failed', 'interrupted'
        """
        self.metadata['status'] = status
        self.metadata['end_timestamp'] = datetime.now().isoformat()
        self.metadata['metrics_summary'] = {
            name: {
                'final': values[-1]['value'] if values else None,
                'min': min(v['value'] for v in values) if values else None,
                'max': max(v['value'] for v in values) if values else None,
                'n_entries': len(values)
            }
            for name, values in self.metrics.items()
        }
        self.metadata['artifacts'] = self.artifacts

        # Save metadata
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save metrics
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        self.logger.info(f"Experiment finished with status: {status}")

    def get_output_dir(self) -> Path:
        """Return the experiment output directory."""
        return self.output_dir


#-------------------------------------------------------
# Reproducibility Utilities
#-------------------------------------------------------

class ReproducibilityManager:
    """
    Manage random seeds and ensure reproducibility across runs.
    """

    def __init__(self, master_seed: int = 42):
        """
        Initialize reproducibility manager.

        Parameters:
        -----------
        master_seed : int
            Master random seed
        """
        self.master_seed = master_seed
        self.seed_history = []
        self._set_seed(master_seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seed for all relevant libraries."""
        np.random.seed(seed)

        # Try to set seeds for other libraries if available
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

        self.seed_history.append({
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        })

    def reset(self) -> None:
        """Reset to master seed."""
        self._set_seed(self.master_seed)

    def set_seed(self, seed: int) -> None:
        """Set a new seed."""
        self._set_seed(seed)

    def get_derived_seed(self, name: str) -> int:
        """
        Get a reproducible derived seed based on a name.

        Parameters:
        -----------
        name : str
            Name to derive seed from

        Returns:
        --------
        int
            Derived seed
        """
        # Create deterministic seed from master seed and name
        combined = f"{self.master_seed}_{name}"
        hash_value = int(hashlib.md5(combined.encode()).hexdigest(), 16)
        return hash_value % (2**31)

    def get_state(self) -> Dict:
        """Get current random state."""
        return {
            'numpy': np.random.get_state(),
            'master_seed': self.master_seed,
            'history': self.seed_history
        }

    def set_state(self, state: Dict) -> None:
        """Restore random state."""
        if 'numpy' in state:
            np.random.set_state(state['numpy'])
        self.master_seed = state.get('master_seed', self.master_seed)


def compute_data_hash(X: np.ndarray) -> str:
    """
    Compute a hash of data for verification.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix

    Returns:
    --------
    str
        MD5 hash of the data
    """
    return hashlib.md5(X.tobytes()).hexdigest()


def verify_data_integrity(X: np.ndarray, expected_hash: str) -> bool:
    """
    Verify data integrity by comparing hashes.

    Parameters:
    -----------
    X : np.ndarray
        Data to verify
    expected_hash : str
        Expected hash value

    Returns:
    --------
    bool
        True if data matches expected hash
    """
    return compute_data_hash(X) == expected_hash


#-------------------------------------------------------
# Convenience Functions
#-------------------------------------------------------

def create_experiment(
    name: str,
    config: Union[str, Dict, SimulationConfig] = None,
    seed: int = 42
) -> Tuple[ExperimentTracker, SimulationConfig, ReproducibilityManager]:
    """
    Create a new experiment with tracking and reproducibility.

    Parameters:
    -----------
    name : str
        Experiment name
    config : str, dict, or SimulationConfig, optional
        Configuration (file path, dictionary, or SimulationConfig object)
    seed : int
        Random seed

    Returns:
    --------
    Tuple of (ExperimentTracker, SimulationConfig, ReproducibilityManager)
    """
    # Setup reproducibility
    repro = ReproducibilityManager(seed)

    # Load configuration
    if config is None:
        sim_config = SimulationConfig()
    elif isinstance(config, str):
        sim_config = SimulationConfig.from_file(config)
    elif isinstance(config, dict):
        sim_config = SimulationConfig(config)
    else:
        sim_config = config

    # Update seed in config
    sim_config.set('simulation', 'random_seed', value=seed)

    # Create tracker
    tracker = ExperimentTracker(name)
    tracker.log_config(sim_config)

    return tracker, sim_config, repro
