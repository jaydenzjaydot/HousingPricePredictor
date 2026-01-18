"""
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
