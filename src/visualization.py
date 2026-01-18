"""
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
