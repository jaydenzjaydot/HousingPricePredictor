"""
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
    file_path = Path(file_path).resolve()
    
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
    
    print("=" * 70)



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
        print(f"\n Columns with missing values:")
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
        print(f"\n 📊 Shape: {original_shape} → {df.shape}")
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
    print("\n Example usage:")
    print("from data import load_data, save_data, get_data_summary")
    print("df = load_data('data/raw/my_file.csv')")
    print("get_data_summary(df)")
    print("df_clean = quick_clean(df)")
    print("save_data(df_clean, 'data/processed/clean_data.csv')")
    print("\n For full documentation, see function docstrings.")
    print("=" * 70)
