#!/usr/bin/env python3
"""
Data Science Project Structure Generator
=========================================
Run this script to automatically create a standardized data science 
project structure with all necessary directories and starter files.

Usage:
    python setup_project.py                    # Creates in current directory
    python setup_project.py --name my_project  # Creates a new project folder
"""

import os
import argparse
from pathlib import Path


def create_directory(path: Path) -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"  📁 Created: {path}")


def create_file(path: Path, content: str = "") -> None:
    """Create a file with optional content if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  📄 Created: {path}")
    else:
        print(f"  ⏭️  Skipped (exists): {path}")


def setup_project(base_path: Path) -> None:
    """Set up the complete data science project structure."""
    
    print(f"\n🚀 Setting up project at: {base_path}\n")
    
    # =========================================================================
    # DIRECTORIES
    # =========================================================================
    directories = [
        "data/raw",           # Original, immutable data
        "data/processed",     # Cleaned data ready for analysis
        "data/external",      # Third-party data sources
        "notebooks",          # Jupyter notebooks
        "src",                # Source code
        "models",             # Saved models and configs
        "reports/figures",    # Generated graphics and reports
        "tests",              # Unit tests
    ]
    
    print("📂 Creating directories...")
    for dir_path in directories:
        create_directory(base_path / dir_path)
    
    # =========================================================================
    # STARTER FILES
    # =========================================================================
    print("\n📝 Creating starter files...")
    
    # ----- requirements.txt -----
    requirements_content = """# Core Data Science Libraries
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.3.0

# Jupyter
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0

# Data Validation & Testing
pytest>=7.4.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.66.0
"""
    create_file(base_path / "requirements.txt", requirements_content)
    
    # ----- environment.yml -----
    env_content = f"""name: {base_path.name}
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.10
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
"""
    create_file(base_path / "environment.yml", env_content)
    
    # ----- README.md -----
    readme_content = f"""# {base_path.name}

## Project Description

[Add your project description here]

## Project Structure

```
{base_path.name}/
├── data/
│   ├── raw/                    # Original, immutable data
│   ├── processed/              # Cleaned data ready for analysis
│   └── external/               # Third-party data sources
├── notebooks/                  # Jupyter notebooks for exploration
├── src/                        # Source code modules
├── models/                     # Trained models and configs
├── reports/figures/            # Generated graphics and reports
├── tests/                      # Unit tests
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
conda activate {base_path.name}
```

### Usage

1. Place raw data in `data/raw/`
2. Run notebooks in order (01, 02, 03, etc.)
3. Find results in `reports/`

## License

[Add your license here]
"""
    create_file(base_path / "README.md", readme_content)
    
    # ----- .gitignore -----
    gitignore_content = """# Data files (often too large for git)
data/raw/*
data/processed/*
data/external/*
!data/*/.gitkeep

# Models
models/*.pkl
models/*.joblib
models/*.h5

# Jupyter checkpoints
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
*.swo

# OS
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local

# Reports (optional - uncomment if needed)
# reports/*.pdf
"""
    create_file(base_path / ".gitignore", gitignore_content)
    
    # ----- LICENSE -----
    license_content = """MIT License

Copyright (c) 2026

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
    create_file(base_path / "LICENSE", license_content)
    
    # ----- src/__init__.py -----
    init_content = '''"""
Source code package for the data science project.
"""
'''
    create_file(base_path / "src/__init__.py", init_content)
    
    # ----- src/data_processing.py -----
    data_processing_content = '''"""
Data Processing Module
======================
Functions for loading, cleaning, and transforming data.
"""

"""
data.py - Centralized Data Management Utilities
================================================

This module provides reusable functions for loading, cleaning, and managing data
across the entire project. Import this in any notebook or script.

Usage:
    from data import load_data, save_data, clean_data
    
    df = load_data('data/raw/my_file.csv')
    df_clean = clean_data(df)
    save_data(df_clean, 'data/processed/clean_data.csv')
"""

import pandas as pd
import numpy as np
import chardet
import os
from pathlib import Path
from typing import Optional, Union, Dict, List
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def detect_encoding(file_path: str, sample_size: int = 100000) -> Dict[str, Union[str, float]]:
    """
    Detect the encoding of a file using chardet.
    
    Args:
        file_path: Path to the file
        sample_size: Number of bytes to read for detection (default: 100KB)
    
    Returns:
        Dictionary with 'encoding' and 'confidence' keys
    
    Example:
        result = detect_encoding('data/file.csv')
        print(f"Encoding: {result['encoding']}, Confidence: {result['confidence']}")
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(sample_size)
        result = chardet.detect(raw_data)
    
    return {
        'encoding': result['encoding'],
        'confidence': result['confidence']
    }


def load_data(
    file_path: str,
    encoding: Optional[str] = None,
    auto_detect: bool = True,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Smart CSV loader with automatic encoding detection.
    
    Args:
        file_path: Path to CSV file (relative or absolute)
        encoding: Specific encoding to use (if None, will auto-detect)
        auto_detect: Whether to auto-detect encoding using chardet
        verbose: Print loading information
        **kwargs: Additional arguments passed to pd.read_csv()
    
    Returns:
        pandas DataFrame
    
    Example:
        # Auto-detect encoding
        df = load_data('data/raw/sales.csv')
        
        # Specify encoding
        df = load_data('data/raw/sales.csv', encoding='latin-1')
        
        # With additional pandas arguments
        df = load_data('data/raw/sales.csv', sep=';', decimal=',')
    """
    # Convert to Path object for better path handling
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect encoding if not specified
    if encoding is None and auto_detect:
        try:
            detected = detect_encoding(str(file_path))
            encoding = detected['encoding']
            if verbose:
                print(f"📂 File: {file_path.name}")
                print(f"🔍 Detected encoding: {encoding} (confidence: {detected['confidence']*100:.1f}%)")
        except Exception as e:
            if verbose:
                print(f"⚠️ Auto-detection failed: {e}")
                print("   Trying common encodings...")
            encoding = None
    
    # Try loading with detected/specified encoding
    if encoding:
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            if verbose:
                print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            if verbose:
                print(f"❌ Failed with {encoding}: {str(e)[:50]}")
    
    # Fallback: Try common encodings
    encodings_to_try = ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8', 'utf-16']
    
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc, **kwargs)
            if verbose:
                print(f"✅ Successfully loaded with {enc} encoding")
                print(f"   {len(df):,} rows and {len(df.columns)} columns")
            return df
        except:
            continue
    
    # Last resort: ignore errors
    try:
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore', **kwargs)
        if verbose:
            print("⚠️ Loaded with UTF-8 (some characters may be lost)")
            print(f"   {len(df):,} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        raise Exception(f"Could not load file with any encoding method: {e}")


def load_excel(
    file_path: str,
    sheet_name: Union[str, int] = 0,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Load Excel file with error handling.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name or index (default: 0)
        verbose: Print loading information
        **kwargs: Additional arguments passed to pd.read_excel()
    
    Returns:
        pandas DataFrame
    
    Example:
        df = load_excel('data/raw/sales.xlsx', sheet_name='Q1')
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        if verbose:
            print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns from {file_path.name}")
        return df
    except Exception as e:
        raise Exception(f"Error loading Excel file: {e}")


def load_multiple_files(
    folder_path: str,
    pattern: str = "*.csv",
    combine: bool = True,
    verbose: bool = True
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load multiple CSV files from a folder.
    
    Args:
        folder_path: Path to folder containing files
        pattern: File pattern to match (default: '*.csv')
        combine: If True, concatenate all files into one DataFrame
        verbose: Print loading information
    
    Returns:
        Single DataFrame if combine=True, else list of DataFrames
    
    Example:
        # Load and combine all CSVs
        df = load_multiple_files('data/raw/monthly_sales/', pattern='sales_*.csv')
        
        # Load as separate DataFrames
        dfs = load_multiple_files('data/raw/', combine=False)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    files = list(folder.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {folder}")
    
    if verbose:
        print(f"📁 Found {len(files)} files matching '{pattern}'")
    
    dataframes = []
    for file in files:
        if verbose:
            print(f"   Loading: {file.name}")
        df = load_data(str(file), verbose=False)
        dataframes.append(df)
    
    if combine:
        combined_df = pd.concat(dataframes, ignore_index=True)
        if verbose:
            print(f"✅ Combined into {len(combined_df):,} rows and {len(combined_df.columns)} columns")
        return combined_df
    else:
        return dataframes


# ============================================================================
# DATA SAVING FUNCTIONS
# ============================================================================

def save_data(
    df: pd.DataFrame,
    file_path: str,
    encoding: str = 'utf-8',
    create_dirs: bool = True,
    verbose: bool = True,
    **kwargs
) -> None:
    """
    Save DataFrame to CSV with automatic directory creation.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        encoding: Encoding to use (default: 'utf-8')
        create_dirs: Create parent directories if they don't exist
        verbose: Print saving information
        **kwargs: Additional arguments passed to df.to_csv()
    
    Example:
        save_data(df, 'data/processed/clean_sales.csv')
    """
    file_path = Path(file_path)
    
    # Create parent directories if needed
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_csv(file_path, encoding=encoding, index=False, **kwargs)
        if verbose:
            print(f"✅ Saved {len(df):,} rows to {file_path}")
    except Exception as e:
        raise Exception(f"Error saving file: {e}")


def save_excel(
    df: pd.DataFrame,
    file_path: str,
    sheet_name: str = 'Sheet1',
    create_dirs: bool = True,
    verbose: bool = True
) -> None:
    """
    Save DataFrame to Excel file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        sheet_name: Name of the Excel sheet
        create_dirs: Create parent directories if they don't exist
        verbose: Print saving information
    
    Example:
        save_excel(df, 'data/processed/sales_report.xlsx', sheet_name='Q1 Sales')
    """
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_excel(file_path, sheet_name=sheet_name, index=False)
        if verbose:
            print(f"✅ Saved {len(df):,} rows to {file_path}")
    except Exception as e:
        raise Exception(f"Error saving Excel file: {e}")


# ============================================================================
# DATA QUALITY FUNCTIONS
# ============================================================================

def get_data_summary(df: pd.DataFrame, show_samples: bool = True) -> None:
    """
    Print comprehensive data summary.
    
    Args:
        df: DataFrame to summarize
        show_samples: Whether to show sample values
    
    Example:
        get_data_summary(df)
    """
    print("=" * 70)
    print("📊 DATA SUMMARY")
    print("=" * 70)
    
    print(f"\n📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
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
    
    print("\n📋 data samples:")
    print("-" * 70)
    
    print(df.head())
    print("\n" + "=" * 70)
    

def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
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
        figsize: Figure size per row (width, height)
        bins: Number of bins for histograms
        show_stats: Display statistics (mean, median, std) on the plot
        save_path: If provided, saves the figure to this path
    
    Example:
        # Plot all numeric columns
        plot_distributions(df)
        
        # Plot specific columns
        plot_distributions(df, columns=['price', 'area', 'bedrooms'])
        
        # Save to file
        plot_distributions(df, save_path='reports/figures/distributions.png')
    """
    # Get numeric columns if not specified
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
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
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()
    print("=" * 70)


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Comprehensive data quality check.
    
    Args:
        df: DataFrame to check
    
    Returns:
        Dictionary with quality metrics
    
    Example:
        quality = check_data_quality(df)
        print(quality['missing_summary'])
    """
    quality_report = {
        'shape': df.shape,
        'total_cells': df.shape[0] * df.shape[1],
        'duplicates': df.duplicated().sum(),
        'missing_summary': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    print("🔍 DATA QUALITY REPORT")
    print("=" * 70)
    print(f"Shape: {quality_report['shape']}")
    print(f"Duplicates: {quality_report['duplicates']:,}")
    print(f"Total Missing Values: {sum(quality_report['missing_summary'].values()):,}")
    print(f"Memory Usage: {quality_report['memory_mb']:.2f} MB")
    
    # Columns with missing values
    missing_cols = {k: v for k, v in quality_report['missing_summary'].items() if v > 0}
    if missing_cols:
        print(f"\nColumns with missing values:")
        for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True):
            pct = (count / df.shape[0]) * 100
            print(f"  - {col}: {count:,} ({pct:.1f}%)")
    
    print("=" * 70)
    
    return quality_report


# ============================================================================
# QUICK CLEANING FUNCTIONS
# ============================================================================

def quick_clean(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    handle_missing: str = 'drop',
    convert_dates: Optional[List[str]] = None,
    convert_cat_cols: Optional[List[str]] = None,
    remove_outliers: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Quick data cleaning with common operations.
    
    Args:
        df: DataFrame to clean
        remove_duplicates: Remove duplicate rows
        handle_missing: 'drop', 'ffill', 'bfill', 'mean', or 'median'
        convert_dates: List of column names to convert to datetime
        convert_cat_cols: List of column names to convert to category type
        remove_outliers: List of numeric columns to remove outliers from (using IQR method)
        verbose: Print cleaning information
    
    Returns:
        Cleaned DataFrame
    
    Example:
        df_clean = quick_clean(
            df, 
            handle_missing='mean', 
            convert_dates=['order_date', 'ship_date'],
            convert_cat_cols=['category', 'region'],
            remove_outliers=['sales', 'profit']
        )
    """
    df = df.copy()
    original_shape = df.shape
    
    if verbose:
        print("🧹 QUICK CLEANING")
        print("=" * 70)
    
    # 1. Remove duplicates
    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if verbose and removed > 0:
            print(f"✓ Removed {removed:,} duplicate rows")
        elif verbose:
            print(f"✓ No duplicates found")
    
    # 2. Handle missing values
    missing_before = df.isnull().sum().sum()
    
    if handle_missing == 'drop':
        before = len(df)
        df = df.dropna()
        removed = before - len(df)
        if verbose and removed > 0:
            print(f"✓ Dropped {removed:,} rows with missing values")
        elif verbose and missing_before > 0:
            print(f"✓ No rows dropped (no missing values)")
            
    elif handle_missing == 'ffill':
        df = df.fillna(method='ffill')
        if verbose:
            print(f"✓ Forward filled {missing_before:,} missing values")
            
    elif handle_missing == 'bfill':
        df = df.fillna(method='bfill')
        if verbose:
            print(f"✓ Backward filled {missing_before:,} missing values")
            
    elif handle_missing == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        if verbose:
            print(f"✓ Filled numeric columns with mean")
            
    elif handle_missing == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        if verbose:
            print(f"✓ Filled numeric columns with median")
    
    # 3. Convert date columns
    if convert_dates:
        for col in convert_dates:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if verbose:
                        print(f"✓ Converted '{col}' to datetime")
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Could not convert '{col}' to datetime: {e}")
            elif verbose:
                print(f"⚠️ Column '{col}' not found for date conversion")
    
    # 4. Convert categorical columns
    if convert_cat_cols:
        for col in convert_cat_cols:
            if col in df.columns:
                try:
                    df[col] = df[col].astype('category')
                    if verbose:
                        print(f"✓ Converted '{col}' to category type")
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Could not convert '{col}' to category: {e}")
            elif verbose:
                print(f"⚠️ Column '{col}' not found for categorical conversion")
    
    # 5. Remove outliers using IQR method
    if remove_outliers:
        for col in remove_outliers:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    before_outliers = len(df)
                    
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    
                    removed_outliers = before_outliers - len(df)
                    if verbose:
                        print(f"✓ Removed {removed_outliers:,} outliers from '{col}'")
                else:
                    if verbose:
                        print(f"⚠️ Column '{col}' is not numeric, skipping outlier removal")
            elif verbose:
                print(f"⚠️ Column '{col}' not found for outlier removal")
    
    if verbose:
        print(f"\n📊 Shape: {original_shape} → {df.shape}")
        rows_removed = original_shape[0] - df.shape[0]
        pct_removed = (rows_removed / original_shape[0]) * 100 if original_shape[0] > 0 else 0
        print(f"   Removed: {rows_removed:,} rows ({pct_removed:.1f}%)")
        print("=" * 70)
    
    return df

# ============================================================================
# PROJECT SETUP FUNCTION
# ============================================================================

def setup_project_structure(base_path: str = '.') -> None:
    """
    Create standard data science project folder structure.
    
    Args:
        base_path: Base directory for the project (default: current directory)
    
    Example:
        setup_project_structure()
    """
    base = Path(base_path)
    
    folders = [
        'data/raw',
        'data/processed',
        'data/external',
        'notebooks',
        'src',
        'models',
        'reports/figures'
    ]
    
    print("📁 Creating project structure...")
    for folder in folders:
        folder_path = base / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {folder}/")
    
    print("\n✅ Project structure created!")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage when running data.py directly
    print("=" * 70)
    print("DATA.PY - Data Management Utilities")
    print("=" * 70)
    print("\nExample usage:\n")
    print("from data import load_data, save_data, get_data_summary")
    print("df = load_data('data/raw/my_file.csv')")
    print("get_data_summary(df)")
    print("df_clean = quick_clean(df)")
    print("save_data(df_clean, 'data/processed/clean_data.csv')")
    print("\nFor full documentation, see function docstrings.")
    print("=" * 70)
'''
    create_file(base_path / "src/data_processing.py", data_processing_content)
    
    # ----- src/feature_engineering.py -----
    feature_eng_content = '''"""
Feature Engineering Module
==========================
Functions for creating and transforming features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

def create_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Create date-based features from a datetime column.
    
    Args:
        df: DataFrame
        date_col: Name of datetime column
        
    Returns:
        DataFrame with new date features
    """
    df = df.copy()
    
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found")
    
    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract features
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_quarter'] = df[date_col].dt.quarter
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    df[f'{date_col}_is_weekend'] = df[date_col].dt.dayofweek >= 5
    df[f'{date_col}_week'] = df[date_col].dt.isocalendar().week
    
    return df


def create_aggregated_features(
    df: pd.DataFrame,
    group_cols: List[str],
    agg_col: str,
    agg_funcs: List[str] = ['mean', 'sum', 'count']
) -> pd.DataFrame:
    """
    Create aggregated features by grouping.
    
    Args:
        df: DataFrame
        group_cols: Columns to group by
        agg_col: Column to aggregate
        agg_funcs: Aggregation functions
        
    Returns:
        DataFrame with aggregated features
    """
    df = df.copy()
    
    agg_df = df.groupby(group_cols)[agg_col].agg(agg_funcs).reset_index()
    agg_df.columns = group_cols + [f'{agg_col}_{func}' for func in agg_funcs]
    
    df = df.merge(agg_df, on=group_cols, how='left')
    
    return df


def create_ratio_features(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Create ratio features from two columns.
    
    Args:
        df: DataFrame
        numerator_col: Numerator column
        denominator_col: Denominator column
        new_col_name: Name for new column (auto-generated if None)
        
    Returns:
        DataFrame with ratio feature
    """
    df = df.copy()
    
    if new_col_name is None:
        new_col_name = f'{numerator_col}_per_{denominator_col}'
    
    # Avoid division by zero
    df[new_col_name] = df[numerator_col] / df[denominator_col].replace(0, np.nan)
    
    return df


def create_binned_features(
    df: pd.DataFrame,
    col: str,
    bins: List[float],
    labels: Optional[List[str]] = None,
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Create binned/categorical features from continuous columns.
    
    Args:
        df: DataFrame
        col: Column to bin
        bins: Bin edges
        labels: Labels for bins
        new_col_name: Name for new column
        
    Returns:
        DataFrame with binned feature
    """
    df = df.copy()
    
    if new_col_name is None:
        new_col_name = f'{col}_binned'
    
    df[new_col_name] = pd.cut(df[col], bins=bins, labels=labels)
    
    return df
'''
    create_file(base_path / "src/feature_engineering.py", feature_eng_content)
    
    # ----- src/modeling.py -----
    modeling_content = '''"""
Modeling Module
===============
Functions for training and evaluating models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from pathlib import Path


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.
    
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2_score': r2_score(y_test, predictions),
        'predictions': predictions
    }
    
    return metrics


def save_model(model, filepath, metadata=None):
    """
    Save model with metadata.
    
    Args:
        model: Trained model
        filepath: Path to save model
        metadata: Additional metadata dict
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to: {filepath}")


def load_model(filepath):
    """Load saved model."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data.get('metadata', {})

'''
    create_file(base_path / "src/modeling.py", modeling_content)
    
    # ----- src/visualization.py -----
    viz_content = '''"""
Visualization Module
====================
Functions for creating plots and figures.
"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def save_figure(filename, directory='../reports/figures', dpi=300):
    """
    Save current figure with consistent settings.
    
    Args:
        filename: Name of file
        directory: Directory to save in
        dpi: Resolution
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    filepath = directory / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")


def plot_distribution(data, title='Distribution', xlabel='Value', bins=30):
    """Plot distribution with mean and median lines."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.axvline(data.mean(), color='red', linestyle='--', 
                label=f'Mean: {data.mean():.2f}')
    plt.axvline(data.median(), color='green', linestyle='--', 
                label=f'Median: {data.median():.2f}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)


def plot_time_series(df, date_col, value_col, title='Time Series'):
    """Plot time series with trend."""
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[value_col], marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
'''
    create_file(base_path / "src/visualization.py", viz_content)
    
    # ----- tests/test_functions.py -----
    tests_content = '''"""
Unit Tests
==========

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import functions to test
from data import (
    load_data,
    save_data,
    get_data_summary,
    check_data_quality,
    quick_clean,
    detect_encoding
)


class TestLoadData:
    """Test data loading functionality"""
    
    def test_load_csv_file(self, sample_csv_file):
        """Test loading a valid CSV file"""
        df = load_data(sample_csv_file, verbose=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0
    
    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent_file.csv', verbose=False)
    
    def test_load_with_encoding(self, sample_csv_file):
        """Test loading with specific encoding"""
        df = load_data(sample_csv_file, encoding='utf-8', verbose=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_auto_detect_encoding(self, sample_csv_file):
        """Test automatic encoding detection"""
        df = load_data(sample_csv_file, auto_detect=True, verbose=False)
        
        assert isinstance(df, pd.DataFrame)


class TestSaveData:
    """Test data saving functionality"""
    
    def test_save_csv(self, sample_dataframe):
        """Test saving dataframe to CSV"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_file = f.name
        
        try:
            save_data(sample_dataframe, temp_file, verbose=False)
            assert os.path.exists(temp_file)
            
            # Verify we can load it back
            df_loaded = pd.read_csv(temp_file)
            assert len(df_loaded) == len(sample_dataframe)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_save_creates_directory(self, sample_dataframe):
        """Test that save_data creates parent directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'new_folder' / 'data.csv'
            
            save_data(sample_dataframe, str(filepath), verbose=False)
            assert filepath.exists()


class TestDataQuality:
    """Test data quality checking functions"""
    
    def test_check_data_quality(self, sample_dataframe):
        """Test data quality check function"""
        quality = check_data_quality(sample_dataframe)
        
        assert isinstance(quality, dict)
        assert 'shape' in quality
        assert 'duplicates' in quality
        assert 'missing_summary' in quality
        assert quality['shape'] == sample_dataframe.shape
    
    def test_detect_missing_values(self, dataframe_with_missing):
        """Test detection of missing values"""
        quality = check_data_quality(dataframe_with_missing)
        
        total_missing = sum(quality['missing_summary'].values())
        assert total_missing > 0
    
    def test_detect_duplicates(self, dataframe_with_duplicates):
        """Test detection of duplicate rows"""
        quality = check_data_quality(dataframe_with_duplicates)
        
        assert quality['duplicates'] > 0


class TestQuickClean:
    """Test data cleaning functionality"""
    
    def test_remove_duplicates(self, dataframe_with_duplicates):
        """Test duplicate removal"""
        df_clean = quick_clean(
            dataframe_with_duplicates,
            remove_duplicates=True,
            handle_missing='drop',
            verbose=False
        )
        
        assert df_clean.duplicated().sum() == 0
        assert len(df_clean) < len(dataframe_with_duplicates)
    
    def test_handle_missing_drop(self, dataframe_with_missing):
        """Test dropping rows with missing values"""
        df_clean = quick_clean(
            dataframe_with_missing,
            handle_missing='drop',
            verbose=False
        )
        
        assert df_clean.isnull().sum().sum() == 0
    
    def test_handle_missing_mean(self, dataframe_with_missing):
        """Test filling missing values with mean"""
        df_clean = quick_clean(
            dataframe_with_missing,
            handle_missing='mean',
            verbose=False
        )
        
        # Check that numeric columns have no missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        assert df_clean[numeric_cols].isnull().sum().sum() == 0
    
    def test_convert_dates(self, sample_dataframe_with_dates):
        """Test date conversion"""
        df_clean = quick_clean(
            sample_dataframe_with_dates,
            convert_dates=['order_date'],
            verbose=False
        )
        
        assert pd.api.types.is_datetime64_any_dtype(df_clean['order_date'])
    
    def test_convert_categorical(self, sample_dataframe):
        """Test categorical conversion"""
        df_clean = quick_clean(
            sample_dataframe,
            convert_cat_cols=['category'],
            verbose=False
        )
        
        if 'category' in df_clean.columns:
            assert df_clean['category'].dtype.name == 'category'
    
    def test_remove_outliers(self, dataframe_with_outliers):
        """Test outlier removal"""
        original_len = len(dataframe_with_outliers)
        
        df_clean = quick_clean(
            dataframe_with_outliers,
            remove_outliers=['value'],
            verbose=False
        )
        
        assert len(df_clean) < original_len


class TestDetectEncoding:
    """Test encoding detection"""
    
    def test_detect_utf8_encoding(self, utf8_csv_file):
        """Test detection of UTF-8 encoding"""
        result = detect_encoding(utf8_csv_file)
        
        assert isinstance(result, dict)
        assert 'encoding' in result
        assert 'confidence' in result
        assert result['encoding'] is not None
'''
    create_file(base_path / "tests/test_functions.py", tests_content)
    
    # ----- models/model_config.json -----
    model_config_content = """{
    "model_type": "RandomForestClassifier",
    "parameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "training": {
        "test_size": 0.2,
        "cross_validation_folds": 5
    }
}
"""
    create_file(base_path / "models/model_config.json", model_config_content)
    
    # ----- reports/technical_report.md -----
    tech_report_content = """# Technical Report

## Overview

[Describe your project and objectives]

## Data

### Data Sources

- Source 1: [Description]
- Source 2: [Description]

### Data Summary

[Add data statistics and summaries]

## Methodology

[Describe your approach]

## Results

[Add your findings]

## Conclusions

[Summarize key takeaways]
"""
    create_file(base_path / "reports/technical_report.md", tech_report_content)
    
    # ----- .gitkeep files for empty directories -----
    gitkeep_dirs = ["data/raw", "data/processed", "data/external", "reports/figures"]
    for dir_path in gitkeep_dirs:
        create_file(base_path / dir_path / ".gitkeep", "")
    
    # ----- Notebook templates -----
    notebook_template = lambda title, desc: f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {title}\\n",
    "\\n",
    "{desc}"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Standard imports\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Project imports\\n",
    "import sys\\n",
    "sys.path.append('..')\\n",
    "from src.data_processing import load_data, save_data\\n",
    "from src.visualization import save_figure\\n",
    "\\n",
    "%matplotlib inline\\n",
    "sns.set_theme(style='whitegrid')"
   ]
  }},
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "## Load Data"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Load your data here\\n",
    "# df = load_data('your_data.csv')\\n",
    "# df.head()"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.10.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}
'''
    
    notebooks = [
        ("01_data_exploration.ipynb", "Data Exploration", "Initial exploration and understanding of the dataset."),
        ("02_data_cleaning.ipynb", "Data Cleaning", "Clean and preprocess the raw data."),
        ("03_feature_engineering.ipynb", "Feature Engineering", "Create and transform features for modeling."),
        ("04_modeling.ipynb", "Modeling", "Train and tune machine learning models."),
        ("05_evaluation.ipynb", "Evaluation", "Evaluate model performance and generate final results."),
    ]
    
    for filename, title, desc in notebooks:
        create_file(base_path / "notebooks" / filename, notebook_template(title, desc))
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ Project setup complete!")
    print("=" * 60)
    print(f"""
📍 Location: {base_path.absolute()}

🚀 Next steps:
   1. cd {base_path.name}
   2. pip install -r requirements.txt
   3. Start coding in notebooks/01_data_exploration.ipynb
   
Happy coding! 🎉
""")


def main():
    parser = argparse.ArgumentParser(
        description="Create a standardized data science project structure"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Project name (creates new folder). If not specified, uses current directory."
    )
    
    args = parser.parse_args()
    
    if args.name:
        base_path = Path.cwd() / args.name
    else:
        base_path = Path.cwd()
    
    setup_project(base_path)


if __name__ == "__main__":
    main()
