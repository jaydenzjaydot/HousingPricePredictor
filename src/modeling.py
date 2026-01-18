"""
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

