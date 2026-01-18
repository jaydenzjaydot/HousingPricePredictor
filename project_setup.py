#!/usr/bin/env python3
"""
Project Setup Module
====================
A comprehensive, class-based toolkit for data science projects.
Provides reusable classes for project structure, data handling,
feature engineering, modeling, and visualization.

Usage:
    from project_setup import ProjectManager, DataManager, FeatureEngineer, ModelManager, Visualizer
    
    # Create project structure
    project = ProjectManager("my_project")
    project.create_structure()
    
    # Load and process data
    dm = DataManager()
    df = dm.load("data/raw/data.csv")
    df = dm.clean(df)
    dm.save(df, "data/processed/clean.csv")
"""

import os
import json
import pickle
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional imports with graceful fallbacks
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    chardet = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ProjectConfig:
    """Configuration for project structure."""
    name: str
    base_path: Path = field(default_factory=lambda: Path.cwd())
    directories: List[str] = field(default_factory=lambda: [
        "data/raw",
        "data/processed",
        "data/external",
        "data/interim",
        "notebooks",
        "src",
        "models",
        "reports/figures",
        "tests",
        "configs",
        "logs"
    ])
    python_version: str = "3.10"
    author: str = ""
    description: str = ""
    license: str = "MIT"
    
    @property
    def project_path(self) -> Path:
        return self.base_path / self.name


@dataclass
class DataConfig:
    """Configuration for data processing."""
    encoding: str = "utf-8"
    date_format: str = "%Y-%m-%d"
    missing_threshold: float = 0.5
    outlier_method: str = "iqr"
    outlier_factor: float = 1.5


@dataclass  
class ModelConfig:
    """Configuration for model training."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    metrics: List[str] = field(default_factory=lambda: ["rmse", "r2", "mae"])


# =============================================================================
# BASE CLASSES
# =============================================================================

class BaseManager(ABC):
    """Abstract base class for all managers."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log_history: List[str] = []
    
    def log(self, message: str, emoji: str = "📌") -> None:
        """Log a message if verbose mode is enabled."""
        log_entry = f"{emoji} {message}"
        self._log_history.append(log_entry)
        if self.verbose:
            print(log_entry)
    
    def get_log_history(self) -> List[str]:
        """Return the log history."""
        return self._log_history.copy()
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return information about the manager state."""
        pass


# =============================================================================
# PROJECT MANAGER
# =============================================================================

class ProjectManager(BaseManager):
    """Manages project structure creation and configuration."""
    
    def __init__(self, name: str = None, base_path: str = ".", verbose: bool = True):
        super().__init__(verbose)
        self.config = ProjectConfig(
            name=name or Path(base_path).name,
            base_path=Path(base_path)
        )
        self._files_created: List[Path] = []
        self._dirs_created: List[Path] = []
    
    def create_structure(self, include_notebooks: bool = True, 
                        include_tests: bool = True) -> "ProjectManager":
        """Create the complete project structure."""
        self.log(f"Setting up project: {self.config.name}", "🚀")
        
        # Create directories
        self._create_directories()
        
        # Create configuration files
        self._create_config_files()
        
        # Create source files
        self._create_source_files()
        
        if include_notebooks:
            self._create_notebooks()
        
        if include_tests:
            self._create_test_files()
        
        self._create_gitkeep_files()
        
        self.log("Project setup complete!", "✅")
        self._print_summary()
        
        return self
    
    def _create_directories(self) -> None:
        """Create project directories."""
        self.log("Creating directories...", "📂")
        project_path = self.config.project_path
        
        for dir_path in self.config.directories:
            full_path = project_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self._dirs_created.append(full_path)
            self.log(f"  Created: {dir_path}/", "📁")
    
    def _create_file(self, path: Path, content: str) -> None:
        """Create a file with content."""
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self._files_created.append(path)
            self.log(f"  Created: {path.name}", "📄")
        else:
            self.log(f"  Skipped (exists): {path.name}", "⏭️")
    
    def _create_config_files(self) -> None:
        """Create configuration files."""
        self.log("Creating configuration files...", "📝")
        project_path = self.config.project_path
        
        # requirements.txt
        requirements = """# Core Data Science Libraries
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
shap>=0.42.0

# Deep Learning (optional)
# torch>=2.0.0
# tensorflow>=2.13.0

# Jupyter
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0

# Data Validation & Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
chardet>=5.2.0
pyyaml>=6.0.0
joblib>=1.3.0

# Data Versioning (optional)
# dvc>=3.0.0
"""
        self._create_file(project_path / "requirements.txt", requirements)
        
        # environment.yml
        environment = f"""name: {self.config.name}
channels:
  - conda-forge
  - defaults
dependencies:
  - python>={self.config.python_version}
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - jupyter
  - pytest
  - pip
  - pip:
    - python-dotenv
    - tqdm
    - chardet
"""
        self._create_file(project_path / "environment.yml", environment)
        
        # README.md
        readme = f"""# {self.config.name}

## Project Description

{self.config.description or "[Add your project description here]"}

## Project Structure

```
{self.config.name}/
├── data/
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned data ready for analysis
│   ├── interim/                # Intermediate data transformations
│   └── external/               # Third-party data sources
├── notebooks/                  # Jupyter notebooks for exploration
├── src/                        # Source code modules
├── models/                     # Trained models and configs
├── reports/figures/            # Generated graphics and reports
├── tests/                      # Unit tests
├── configs/                    # Configuration files
├── logs/                       # Log files
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
└── README.md
```

## Getting Started

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate {self.config.name}
```

### Usage

```python
from project_setup import ProjectManager, DataManager, FeatureEngineer

# Load data
dm = DataManager()
df = dm.load("data/raw/data.csv")

# Clean and process
df = dm.clean(df)
dm.save(df, "data/processed/clean_data.csv")
```

## License

{self.config.license}
"""
        self._create_file(project_path / "README.md", readme)
        
        # .gitignore
        gitignore = """# Data files
data/raw/*
data/processed/*
data/external/*
data/interim/*
!data/*/.gitkeep

# Models
models/*.pkl
models/*.joblib
models/*.h5
models/*.pt

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Logs
logs/*.log
"""
        self._create_file(project_path / ".gitignore", gitignore)
        
        # LICENSE
        license_text = f"""MIT License

Copyright (c) {datetime.now().year} {self.config.author or "[Author]"}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""
        self._create_file(project_path / "LICENSE", license_text)
        
        # configs/config.yaml
        config_yaml = """# Project Configuration

project:
  name: "{name}"
  version: "0.1.0"

data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  encoding: "utf-8"

model:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""".format(name=self.config.name)
        self._create_file(project_path / "configs/config.yaml", config_yaml)
    
    def _create_source_files(self) -> None:
        """Create source code files."""
        self.log("Creating source files...", "📝")
        src_path = self.config.project_path / "src"
        
        # __init__.py
        init_content = '''"""
Source code package for the data science project.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
'''
        self._create_file(src_path / "__init__.py", init_content)
        
        # utils.py
        utils_content = '''"""
Utility Functions
=================
Common utility functions used across the project.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any
from functools import wraps
from datetime import datetime


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    
    if path.suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        duration = datetime.now() - start
        print(f"⏱️ {func.__name__} took {duration.total_seconds():.2f}s")
        return result
    return wrapper


def ensure_dir(path: str) -> Path:
    """Ensure a directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
'''
        self._create_file(src_path / "utils.py", utils_content)
    
    def _create_notebooks(self) -> None:
        """Create notebook templates."""
        self.log("Creating notebook templates...", "📓")
        notebooks_path = self.config.project_path / "notebooks"
        
        notebooks = [
            ("01_data_exploration.ipynb", "Data Exploration", 
             "Initial exploration and understanding of the dataset."),
            ("02_data_cleaning.ipynb", "Data Cleaning",
             "Clean and preprocess the raw data."),
            ("03_feature_engineering.ipynb", "Feature Engineering",
             "Create and transform features for modeling."),
            ("04_modeling.ipynb", "Modeling",
             "Train and tune machine learning models."),
            ("05_evaluation.ipynb", "Evaluation",
             "Evaluate model performance and generate final results."),
        ]
        
        for filename, title, desc in notebooks:
            notebook = self._create_notebook_json(title, desc)
            self._create_file(notebooks_path / filename, notebook)
    
    def _create_notebook_json(self, title: str, description: str) -> str:
        """Create notebook JSON content."""
        return json.dumps({
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {title}\n", "\n", description]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Standard imports\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "\n",
                        "# Project imports\n",
                        "import sys\n",
                        "sys.path.append('..')\n",
                        "from project_setup import DataManager, FeatureEngineer, Visualizer\n",
                        "\n",
                        "%matplotlib inline\n",
                        "sns.set_theme(style='whitegrid')"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## Load Data"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Load your data\n",
                        "dm = DataManager()\n",
                        "# df = dm.load('../data/raw/your_data.csv')\n",
                        "# df.head()"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }, indent=1)
    
    def _create_test_files(self) -> None:
        """Create test file templates."""
        self.log("Creating test files...", "🧪")
        tests_path = self.config.project_path / "tests"
        
        conftest = '''"""
Pytest Configuration and Fixtures
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": range(1, 101),
        "value": np.random.randn(100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "date": pd.date_range("2024-01-01", periods=100)
    })


@pytest.fixture
def dataframe_with_missing():
    """Create a DataFrame with missing values."""
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5],
        "b": [np.nan, 2, 3, np.nan, 5],
        "c": ["x", "y", None, "z", "w"]
    })
    return df


@pytest.fixture
def sample_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)
'''
        self._create_file(tests_path / "conftest.py", conftest)
        
        test_init = '''"""Tests package."""'''
        self._create_file(tests_path / "__init__.py", test_init)
    
    def _create_gitkeep_files(self) -> None:
        """Create .gitkeep files for empty directories."""
        gitkeep_dirs = [
            "data/raw", "data/processed", "data/external", 
            "data/interim", "reports/figures", "logs"
        ]
        for dir_path in gitkeep_dirs:
            self._create_file(
                self.config.project_path / dir_path / ".gitkeep", ""
            )
    
    def _print_summary(self) -> None:
        """Print project setup summary."""
        print(f"""
{'='*60}
📍 Location: {self.config.project_path.absolute()}

🚀 Next steps:
   1. cd {self.config.name}
   2. pip install -r requirements.txt
   3. Start coding in notebooks/01_data_exploration.ipynb
   
Happy coding! 🎉
{'='*60}
""")
    
    def get_info(self) -> Dict[str, Any]:
        """Return project information."""
        return {
            "name": self.config.name,
            "path": str(self.config.project_path),
            "directories_created": len(self._dirs_created),
            "files_created": len(self._files_created)
        }


# =============================================================================
# DATA MANAGER
# =============================================================================

class DataManager(BaseManager):
    """Manages data loading, saving, and processing operations."""
    
    def __init__(self, config: DataConfig = None, verbose: bool = True):
        super().__init__(verbose)
        self.config = config or DataConfig()
        self._loaded_files: Dict[str, pd.DataFrame] = {}
    
    # -------------------------------------------------------------------------
    # Loading Methods
    # -------------------------------------------------------------------------
    
    def detect_encoding(self, file_path: str, sample_size: int = 100000) -> Dict[str, Any]:
        """Detect file encoding using chardet."""
        if not HAS_CHARDET:
            self.log("chardet not installed, using utf-8 as default", "⚠️")
            return {"encoding": "utf-8", "confidence": 1.0}
        
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
        return {"encoding": result["encoding"], "confidence": result["confidence"]}
    
    def load(self, file_path: str, encoding: str = None, 
             auto_detect: bool = True, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV with automatic encoding detection.
        
        Args:
            file_path: Path to the file
            encoding: Specific encoding (auto-detected if None)
            auto_detect: Whether to auto-detect encoding
            **kwargs: Additional pandas read_csv arguments
        
        Returns:
            Loaded DataFrame
        """

        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect encoding
        if encoding is None and auto_detect:
            try:
                detected = self.detect_encoding(str(file_path))
                encoding = detected["encoding"]
                self.log(f"File: {file_path.name}", "📂")
                self.log(f"Detected encoding: {encoding} ({detected['confidence']*100:.1f}%)", "🔍")
                

            except Exception as e:
                self.log(f"Auto-detection failed: {e}", "⚠️")
                encoding = None
        
        # Try loading with detected encoding
        if encoding:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                self.log(f"Loaded {len(df):,} rows and {len(df.columns)} columns", "✅")
                self.log(f"Column names:", "📋")
                for i, col in enumerate(df.columns, 1):
                    self.log(f"  {i:2d}. {col}", "")
                self._loaded_files[str(file_path)] = df
                return df
            except Exception as e:
                self.log(f"Failed with {encoding}: {str(e)[:50]}", "❌")
        
        # Fallback encodings
        for enc in ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8', 'utf-16']:
            try:
                df = pd.read_csv(file_path, encoding=enc, **kwargs)
                self.log(f"Loaded with {enc} encoding", "✅")
                self.log(f"Loaded {len(df):,} rows and {len(df.columns)} columns", "✅")
                self.log(f"Column names:", "📋")
                for i, col in enumerate(df.columns, 1):
                    self.log(f"  {i:2d}. {col}", "")
                self._loaded_files[str(file_path)] = df
                return df
            except:
                continue
        
        raise Exception(f"Could not load file: {file_path}")
        


    def load_excel(self, file_path: str, sheet_name: Union[str, int] = 0,
                   **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        self.log(f"Loaded {len(df):,} rows from {file_path.name}", "✅")
        return df
    
    def load_multiple(self, folder_path: str, pattern: str = "*.csv",
                      combine: bool = True) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Load multiple files from a folder."""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        files = list(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {folder}")
        
        self.log(f"Found {len(files)} files matching '{pattern}'", "📁")
        
        dataframes = []
        for file in files:
            self.log(f"  Loading: {file.name}", "📄")
            df = self.load(str(file), verbose=False)
            dataframes.append(df)
        
        if combine:
            combined = pd.concat(dataframes, ignore_index=True)
            self.log(f"Combined: {len(combined):,} rows", "✅")
            return combined
        return dataframes
    
    # -------------------------------------------------------------------------
    # Saving Methods
    # -------------------------------------------------------------------------
    
    def save(self, df: pd.DataFrame, file_path: str, 
             encoding: str = "utf-8", **kwargs) -> None:
        """Save DataFrame to CSV."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(file_path, encoding=encoding, index=False, **kwargs)
        self.log(f"Saved {len(df):,} rows to {file_path}", "✅")
    
    def save_excel(self, df: pd.DataFrame, file_path: str,
                   sheet_name: str = "Sheet1") -> None:
        """Save DataFrame to Excel."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
        self.log(f"Saved {len(df):,} rows to {file_path}", "✅")
    
    # -------------------------------------------------------------------------
    # Data Quality Methods
    # -------------------------------------------------------------------------
    
    def summary(self, df: pd.DataFrame, show_samples: bool = True) -> Dict[str, Any]:
        """Get comprehensive data summary."""
        print("=" * 70)
        print("📊 DATA SUMMARY")
        print("=" * 70)
        
        print(f"\n📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📋 Column Information:")
        print("-" * 70)
        
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique = df[col].nunique()
            
            print(f"{i:2d}. {col:30s} | {str(dtype):10s} | "
                  f"Nulls: {null_count:6,} ({null_pct:5.1f}%) | "
                  f"Unique: {unique:,}")
            
            if show_samples and unique <= 10:
                samples = df[col].dropna().unique()[:5]
                print(f"    Samples: {list(samples)}")
        
        print("\n📋 First rows:")
        print("-" * 70)
        print(df.head())
        print("=" * 70)
        
        return {
            "shape": df.shape,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "dtypes": df.dtypes.value_counts().to_dict()
        }
    
    def quality_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality check."""
        report = {
            "shape": df.shape,
            "total_cells": df.shape[0] * df.shape[1],
            "duplicates": df.duplicated().sum(),
            "missing": df.isnull().sum().to_dict(),
            "missing_total": df.isnull().sum().sum(),
            "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "dtypes": df.dtypes.value_counts().to_dict(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        print("🔍 DATA QUALITY REPORT")
        print("=" * 70)
        print(f"Shape: {report['shape']}")
        print(f"Duplicates: {report['duplicates']:,}")
        print(f"Missing Values: {report['missing_total']:,} ({report['missing_pct']:.1f}%)")
        print(f"Memory: {report['memory_mb']:.2f} MB")
        
        missing_cols = {k: v for k, v in report["missing"].items() if v > 0}
        if missing_cols:
            print("\nColumns with missing values:")
            for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True):
                pct = (count / df.shape[0]) * 100
                print(f"  - {col}: {count:,} ({pct:.1f}%)")
        
        print("=" * 70)
        return report
    
    # -------------------------------------------------------------------------
    # Cleaning Methods
    # -------------------------------------------------------------------------
    
    def quick_clean(self, df: pd.DataFrame, 
              remove_duplicates: bool = True,
              handle_missing: str = "drop",
              convert_dates: List[str] = None,
              convert_categories: List[str] = None,
              remove_outliers: List[str] = None,
              save_path: str = None) -> pd.DataFrame:
        """
        Perform data cleaning operations.
        
        Args:
            df: Input DataFrame
            remove_duplicates: Remove duplicate rows
            handle_missing: 'drop', 'ffill', 'bfill', 'mean', 'median'
            convert_dates: Columns to convert to datetime
            convert_categories: Columns to convert to category
            remove_outliers: Columns to remove outliers from (IQR method)
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        original_shape = df.shape
        
        self.log("CLEANING DATA", "🧹")
        print("=" * 70)
        
        # Remove duplicates
        if remove_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            if removed > 0:
                self.log(f"Removed {removed:,} duplicate rows", "✓")
        
        # Handle missing values
        if handle_missing == "drop":
            before = len(df)
            df = df.dropna()
            self.log(f"Dropped {before - len(df):,} rows with missing values", "✓")
        elif handle_missing == "ffill":
            df = df.fillna(method="ffill")
            self.log("Forward filled missing values", "✓")
        elif handle_missing == "bfill":
            df = df.fillna(method="bfill")
            self.log("Backward filled missing values", "✓")
        elif handle_missing == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
            self.log("Filled numeric columns with mean", "✓")
        elif handle_missing == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
            self.log("Filled numeric columns with median", "✓")
        
        # Convert dates
        if convert_dates:
            for col in convert_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    self.log(f"Converted '{col}' to datetime", "✓")
        
        # Convert categories
        if convert_categories:
            for col in convert_categories:
                if col in df.columns:
                    df[col] = df[col].astype("category")
                    self.log(f"Converted '{col}' to category", "✓")
        
        # Remove outliers
        if remove_outliers:
            for col in remove_outliers:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    before = len(df)
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
                    self.log(f"Removed {before - len(df):,} outliers from '{col}'", "✓")
        
        # Save cleaned data
        if save_path:
            self.save(df, save_path)
            
        
        # Summary
        rows_removed = original_shape[0] - df.shape[0]
        pct_removed = (rows_removed / original_shape[0]) * 100 if original_shape[0] > 0 else 0
        print(f"\n📊 Shape: {original_shape} → {df.shape}")
        print(f"   Removed: {rows_removed:,} rows ({pct_removed:.1f}%)")
        print("=" * 70)
        
        return df

    
    def get_info(self) -> Dict[str, Any]:
        """Return data manager state."""
        return {
            "loaded_files": list(self._loaded_files.keys()),
            "config": self.config.__dict__
        }


# =============================================================================
# FEATURE ENGINEER
# =============================================================================

class FeatureEngineer(BaseManager):
    """Manages feature creation and transformation."""
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self._transformations: List[str] = []
    
    def date_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Extract date-based features from a datetime column."""
        df = df.copy()
        
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        df[f"{date_col}_year"] = df[date_col].dt.year
        df[f"{date_col}_month"] = df[date_col].dt.month
        df[f"{date_col}_quarter"] = df[date_col].dt.quarter
        df[f"{date_col}_day"] = df[date_col].dt.day
        df[f"{date_col}_dayofweek"] = df[date_col].dt.dayofweek
        df[f"{date_col}_is_weekend"] = df[date_col].dt.dayofweek >= 5
        df[f"{date_col}_week"] = df[date_col].dt.isocalendar().week
        
        self.log(f"Created 7 date features from '{date_col}'", "✅")
        self._transformations.append(f"date_features({date_col})")
        
        return df
    
    def aggregated_features(self, df: pd.DataFrame, group_cols: List[str],
                           agg_col: str, agg_funcs: List[str] = None) -> pd.DataFrame:
        """Create aggregated features by grouping."""
        df = df.copy()
        agg_funcs = agg_funcs or ["mean", "sum", "count"]
        
        agg_df = df.groupby(group_cols)[agg_col].agg(agg_funcs).reset_index()
        agg_df.columns = group_cols + [f"{agg_col}_{func}" for func in agg_funcs]
        
        df = df.merge(agg_df, on=group_cols, how="left")
        
        self.log(f"Created {len(agg_funcs)} aggregated features for '{agg_col}'", "✅")
        return df
    
    def ratio_feature(self, df: pd.DataFrame, numerator: str, 
                      denominator: str, name: str = None) -> pd.DataFrame:
        """Create ratio feature from two columns."""
        df = df.copy()
        name = name or f"{numerator}_per_{denominator}"
        df[name] = df[numerator] / df[denominator].replace(0, np.nan)
        
        self.log(f"Created ratio feature '{name}'", "✅")
        return df
    
    def binned_feature(self, df: pd.DataFrame, col: str, bins: List[float],
                       labels: List[str] = None, name: str = None) -> pd.DataFrame:
        """Create binned/categorical feature from continuous column."""
        df = df.copy()
        name = name or f"{col}_binned"
        df[name] = pd.cut(df[col], bins=bins, labels=labels)
        
        self.log(f"Created binned feature '{name}'", "✅")
        return df
    
    def polynomial_features(self, df: pd.DataFrame, columns: List[str],
                           degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for specified columns."""
        df = df.copy()
        new_cols = 0
        
        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f"{col}_pow{d}"] = df[col] ** d
                    new_cols += 1
        
        self.log(f"Created {new_cols} polynomial features (degree {degree})", "✅")
        return df
    
    def interaction_features(self, df: pd.DataFrame, 
                            col_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between column pairs."""
        df = df.copy()
        
        for col1, col2 in col_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        
        self.log(f"Created {len(col_pairs)} interaction features", "✅")
        return df
    
    def log_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply log transformation to specified columns."""
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
        
        self.log(f"Created {len(columns)} log-transformed features", "✅")
        return df
    
    def normalize(self, df: pd.DataFrame, columns: List[str] = None,
                  method: str = "standard") -> pd.DataFrame:
        """Normalize numeric columns."""
        df = df.copy()
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                if method == "standard":
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
                elif method == "minmax":
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        self.log(f"Normalized {len(columns)} columns using {method}", "✅")
        return df
    
    def get_info(self) -> Dict[str, Any]:
        """Return transformation history."""
        return {"transformations": self._transformations}


# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager(BaseManager):
    """Manages model training, evaluation, and persistence."""
    
    def __init__(self, config: ModelConfig = None, verbose: bool = True):
        super().__init__(verbose)
        self.config = config or ModelConfig()
        self._models: Dict[str, Any] = {}
        self._metrics: Dict[str, Dict] = {}
    
    def encode_categorical(self, df: pd.DataFrame, 
                            columns: List[str] = None,
                            strategy: str = "label",
                            drop_original: bool = True,
                            handle_unknown: str = "error") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode categorical columns before testing the model.
        
        Args:
            df: Input DataFrame containing categorical columns
            columns: List of columns to encode (if None, auto-detects categorical columns)
            strategy: Encoding strategy - 'label', 'onehot', or 'ordinal'
            drop_original: Whether to drop original categorical columns after encoding
            handle_unknown: How to handle unknown categories during transform - 
                           'error' (raise error), 'ignore' (for onehot only)
        
        Returns:
            Tuple of (encoded DataFrame, dictionary of fitted encoders)
        
        Example:
            >>> model_manager = ModelManager()
            >>> df_encoded, encoders = model_manager.encode_categorical(df, strategy='onehot')
            >>> # Later, transform new data using the same encoders
            >>> new_df_encoded = model_manager.transform_categorical(new_df, encoders)
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
        
        df = df.copy()
        encoders = {}
        
        # Auto-detect categorical columns if not specified
        if columns is None:
            # Select object dtype and category dtype columns
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            self.log(f"Auto-detected {len(columns)} categorical columns", "🔍")
        
        if not columns:
            self.log("No categorical columns to encode", "⚠️")
            return df, encoders
        
        self.log(f"Encoding {len(columns)} categorical columns using '{strategy}' strategy", "🔄")
        
        if strategy == "label":
            # Label Encoding - converts each category to an integer
            for col in columns:
                if col in df.columns:
                    le = LabelEncoder()
                    # Handle NaN values by filling with placeholder
                    mask = df[col].notna()
                    df.loc[mask, f"{col}_encoded"] = le.fit_transform(df.loc[mask, col].astype(str))
                    df[f"{col}_encoded"] = df[f"{col}_encoded"].astype('Int64')
                    encoders[col] = le
                    self.log(f"  Label encoded '{col}' ({len(le.classes_)} classes)", "✓")
                    
                    if drop_original:
                        df = df.drop(columns=[col])
        
        elif strategy == "onehot":
            # One-Hot Encoding - creates binary columns for each category
            for col in columns:
                if col in df.columns:
                    ohe = OneHotEncoder(sparse_output=False, 
                                       handle_unknown='ignore' if handle_unknown == 'ignore' else 'error')
                    
                    # Reshape for OneHotEncoder
                    col_data = df[[col]].fillna('_MISSING_').astype(str)
                    encoded = ohe.fit_transform(col_data)
                    
                    # Create column names
                    categories = ohe.categories_[0]
                    encoded_cols = [f"{col}_{cat}" for cat in categories]
                    
                    # Add encoded columns to DataFrame
                    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
                    df = pd.concat([df, encoded_df], axis=1)
                    
                    encoders[col] = ohe
                    self.log(f"  One-hot encoded '{col}' ({len(categories)} categories)", "✓")
                    
                    if drop_original:
                        df = df.drop(columns=[col])
        
        elif strategy == "ordinal":
            # Ordinal Encoding - similar to label encoding but handles multiple columns
            for col in columns:
                if col in df.columns:
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', 
                                       unknown_value=-1)
                    
                    # Reshape for OrdinalEncoder
                    col_data = df[[col]].fillna('_MISSING_').astype(str)
                    df[f"{col}_encoded"] = oe.fit_transform(col_data).astype(int)
                    
                    encoders[col] = oe
                    self.log(f"  Ordinal encoded '{col}' ({len(oe.categories_[0])} categories)", "✓")
                    
                    if drop_original:
                        df = df.drop(columns=[col])
        
        else:
            raise ValueError(f"Unknown encoding strategy: '{strategy}'. "
                           f"Choose from: 'label', 'onehot', 'ordinal'")
        
        self.log(f"Encoding complete. New shape: {df.shape}", "✅")
        
        # Store encoders for later use
        self._encoders = encoders
        
        return df, encoders
    
    def transform_categorical(self, df: pd.DataFrame, 
                               encoders: Dict[str, Any] = None,
                               drop_original: bool = True) -> pd.DataFrame:
        """
        Transform categorical columns using previously fitted encoders.
        
        Args:
            df: Input DataFrame to transform
            encoders: Dictionary of fitted encoders (if None, uses stored encoders)
            drop_original: Whether to drop original categorical columns
        
        Returns:
            Transformed DataFrame
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
        
        df = df.copy()
        encoders = encoders or getattr(self, '_encoders', {})
        
        if not encoders:
            self.log("No encoders available. Call encode_categorical first.", "⚠️")
            return df
        
        for col, encoder in encoders.items():
            if col not in df.columns:
                self.log(f"Column '{col}' not found in DataFrame, skipping", "⚠️")
                continue
            
            if isinstance(encoder, LabelEncoder):
                mask = df[col].notna()
                try:
                    df.loc[mask, f"{col}_encoded"] = encoder.transform(df.loc[mask, col].astype(str))
                    df[f"{col}_encoded"] = df[f"{col}_encoded"].astype('Int64')
                except ValueError as e:
                    self.log(f"Error transforming '{col}': {e}", "❌")
                    continue
                    
            elif isinstance(encoder, OneHotEncoder):
                col_data = df[[col]].fillna('_MISSING_').astype(str)
                try:
                    encoded = encoder.transform(col_data)
                    categories = encoder.categories_[0]
                    encoded_cols = [f"{col}_{cat}" for cat in categories]
                    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
                    df = pd.concat([df, encoded_df], axis=1)
                except ValueError as e:
                    self.log(f"Error transforming '{col}': {e}", "❌")
                    continue
                    
            elif isinstance(encoder, OrdinalEncoder):
                col_data = df[[col]].fillna('_MISSING_').astype(str)
                try:
                    df[f"{col}_encoded"] = encoder.transform(col_data).astype(int)
                except ValueError as e:
                    self.log(f"Error transforming '{col}': {e}", "❌")
                    continue
            
            if drop_original:
                df = df.drop(columns=[col])
        
        self.log(f"Transformation complete. Shape: {df.shape}", "✅")
        return df

    def train_test_split(self, X: pd.DataFrame, y: pd.Series,
                         test_size: float = None) -> Tuple:
        """Split data into train and test sets."""
        from sklearn.model_selection import train_test_split
        
        test_size = test_size or self.config.test_size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_state
        )
        
        self.log(f"Split data: Train={len(X_train):,}, Test={len(X_test):,}", "✅")
        return X_train, X_test, y_train, y_test
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = "model") -> Dict[str, float]:
        """Evaluate regression model predictions."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        self._metrics[model_name] = metrics
        
        print(f"📊 {model_name} Evaluation:")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE:  {metrics['mae']:.4f}")
        print(f"   R²:   {metrics['r2']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               model_name: str = "model") -> Dict[str, float]:
        """Evaluate classification model predictions."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted")
        }
        
        self._metrics[model_name] = metrics
        
        print(f"📊 {model_name} Evaluation:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def save_model(self, model: Any, filepath: str, metadata: Dict = None) -> None:
        """Save model with metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": model,
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        
        self.log(f"Model saved to: {filepath}", "✅")
    
    def load_model(self, filepath: str) -> Tuple[Any, Dict]:
        """Load saved model."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        
        self.log(f"Model loaded from: {filepath}", "✅")
        return model_data["model"], model_data.get("metadata", {})
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models."""
        if not self._metrics:
            return pd.DataFrame()
        
        return pd.DataFrame(self._metrics).T
    
    def get_info(self) -> Dict[str, Any]:
        """Return model manager state."""
        return {
            "models": list(self._models.keys()),
            "metrics": self._metrics,
            "config": self.config.__dict__
        }


# =============================================================================
# VISUALIZER
# =============================================================================

class Visualizer(BaseManager):
    """Manages data visualization and plotting."""
    
    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (10, 6),
                 verbose: bool = True):
        super().__init__(verbose)
        self.style = style
        self.figsize = figsize
        self._figures: List[str] = []
        sns.set_style(style)
    
    def distribution(self, data: pd.Series, title: str = None,
                     bins: int = 30, save_path: str = None) -> None:
        """Plot distribution with mean and median."""
        plt.figure(figsize=self.figsize)
        
        plt.hist(data.dropna(), bins=bins, edgecolor="black", alpha=0.7)
        plt.axvline(data.mean(), color="red", linestyle="--",
                   label=f"Mean: {data.mean():.2f}")
        plt.axvline(data.median(), color="green", linestyle="--",
                   label=f"Median: {data.median():.2f}")
        
        plt.title(title or f"Distribution of {data.name}")
        plt.xlabel(data.name)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        
        if save_path:
            self._save_figure(save_path)
        
        plt.show()
    
    
    """Plot distributions for multiple columns."""
    def distributions(self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        figsize: tuple = (15, 4),
        bins: int = 30,
        show_stats: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the distribution of numeric columns with histograms and box plots.
        
        Args:
            df: DataFrame to visualize
            columns: List of columns to plot (if None, plots all numeric columns)
            exclude: List of columns to exclude from plotting
            figsize: Figure size per row (width, height)
            bins: Number of bins for histograms
            show_stats: Display statistics (mean, median, std) on the plot
            save_path: If provided, saves the figure to this path
        
        Example:
            # Plot all numeric columns
            distributions(df)
            
            # Plot specific columns
            distributions(df, columns=['price', 'area', 'bedrooms'])
            
            # Exclude specific columns
            distributions(df, exclude=['id', 'zip_code'])
            
            # Save to file
            distributions(df, save_path='reports/figures/distributions.png')
        """
        # Get numeric columns if not specified
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        # Exclude specified columns
        if exclude is not None:
            numeric_cols = [col for col in numeric_cols if col not in exclude]
        
        if not numeric_cols:
            print("⚠️ No numeric columns found to plot.")
            return
        
        n_cols = len(numeric_cols)
        
        print(f"📊 Plotting distributions for {n_cols} numeric columns...")
        print("=" * 70)
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create subplot grid: 2 plots per column (histogram + boxplot)
        fig, axes = plt.subplots(n_cols, 2, figsize=(figsize[0], figsize[1] * n_cols))
        
        # Handle single column case
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(numeric_cols):
            data = df[col].dropna()
            
            # Histogram with KDE
            ax1 = axes[idx, 0]
            sns.histplot(data, bins=bins, kde=True, ax=ax1, color='steelblue', alpha=0.7)
            ax1.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Frequency')
            
            # Add statistics
            if show_stats:
                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()
                
                # Add vertical lines for mean and median
                ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax1.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
                ax1.legend(fontsize=9)
                
                # Add stats text box
                stats_text = f'Std: {std_val:.2f}\nMin: {data.min():.2f}\nMax: {data.max():.2f}'
                ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Box plot
            ax2 = axes[idx, 1]
            sns.boxplot(x=data, ax=ax2, color='steelblue')
            ax2.set_title(f'Box Plot of {col}', fontsize=12, fontweight='bold')
            ax2.set_xlabel(col)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            self._save_figure(save_path)
        
        plt.show()
    
    def correlation_heatmap(self, df: pd.DataFrame, save_path: str = None) -> None:
        """Plot correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="RdBu_r", center=0,
                   fmt=".2f", square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        
        if save_path:
            self._save_figure(save_path)
        
        plt.show()
    
    def scatter_matrix(self, df: pd.DataFrame, columns: List[str] = None,
                       exclude: List[str] = None,
                       save_path: str = None) -> None:
        """Create scatter plot matrix."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns[:5].tolist()
        
        # Exclude specified columns
        if exclude is not None:
            columns = [col for col in columns if col not in exclude]
        
        
        pd.plotting.scatter_matrix(df[columns], figsize=(12, 12), 
                                   diagonal="kde", alpha=0.5)
        plt.suptitle("Scatter Matrix", y=1.02)
        
        if save_path:
            self._save_figure(save_path)
        
        plt.show()
    
    def time_series(self, df: pd.DataFrame, date_col: str, value_col: str,
                    title: str = None, save_path: str = None) -> None:
        """Plot time series."""
        plt.figure(figsize=self.figsize)
        plt.plot(df[date_col], df[value_col], marker="o", markersize=3)
        plt.title(title or f"{value_col} over time")
        plt.xlabel("Date")
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            self._save_figure(save_path)
        
        plt.show()
    
    def feature_importance(self, importance: np.ndarray, feature_names: List[str],results: pd.Dataframe,feature_cols: pd.DataFrame,
                          best_model_name: str = None,top_n: int = 20, save_path: str = None) -> None:
        """Plot feature importance."""
        
        # Get feature importance from best tree-based model
        if 'Random Forest' in results or 'Gradient Boosting' in results:
            importance_model = results.get('Random Forest', results.get('Gradient Boosting'))['model']
            
            if hasattr(importance_model, 'feature_importances_'):
                importances = importance_model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(feature_importance.head(10).to_string(index=False))
                
                # Plot feature importance
                plt.figure(figsize=(10, 6))
                top_features = feature_importance.head(15)
                plt.barh(range(len(top_features)), top_features['Importance'],color = "steelblue")
                plt.yticks(range(len(top_features)), top_features['Feature'])
                plt.xlabel('Importance')
                plt.title(f'Top 15 Feature Importances - {best_model_name}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig('../reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()

    
    def residuals(self, y_test: np.ndarray, y_pred: np.ndarray,best_model_name: str = None,
                  save_path: str = "reports/figures/residuals.png") -> None:
        """Plot residuals for regression model."""
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        
        # Subplot 1: Scatter plot
    
        axes[1].scatter(y_test, y_pred, alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Prices ($)')
        axes[1].set_ylabel('Predicted Prices ($)')
        axes[1].set_title(f'Predictions vs Actual - {best_model_name}')
        axes[1].grid(True, alpha=0.3)
        
        # Residuals distribution
        sns.histplot(residuals, kde=True, ax=axes[2])
        axes[2].set_xlabel("Residuals")
        axes[2].set_title("Residuals Distribution")
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(save_path)
        
        plt.show()
    
    def _save_figure(self, path: str, dpi: int = 150) -> None:
        """Save current figure."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        self._figures.append(str(path))
        self.log(f"Figure saved: {path}", "✅")
    
    def get_info(self) -> Dict[str, Any]:
        """Return visualizer state."""
        return {
            "style": self.style,
            "figsize": self.figsize,
            "figures_saved": self._figures
        }


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    """Command-line interface for project setup."""
    parser = argparse.ArgumentParser(
        description="Create a standardized data science project structure"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Project name (creates new folder)"
    )
    parser.add_argument(
        "--author", "-a",
        type=str,
        default="",
        help="Author name"
    )
    parser.add_argument(
        "--no-notebooks",
        action="store_true",
        help="Skip creating notebook templates"
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip creating test files"
    )
    
    args = parser.parse_args()
    
    # Create project
    project = ProjectManager(name=args.name)
    project.config.author = args.author
    project.create_structure(
        include_notebooks=not args.no_notebooks,
        include_tests=not args.no_tests
    )


if __name__ == "__main__":
    main()
