"""
Utility functions for Causal Forest implementation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples=1000, n_features=10, treatment_effect_coef=None, random_state=None):
    """
    Generate synthetic data for testing causal forest.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features to generate
    treatment_effect_coef : array-like, optional
        Coefficients for heterogeneous treatment effect. If None, random coefficients will be generated.
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X : array-like
        Feature matrix
    t : array-like
        Treatment assignment (0 or 1)
    y : array-like
        Outcome variable
    true_effect : array-like
        True treatment effect for each sample
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Generate propensity scores (probability of treatment)
    propensity_score = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    
    # Assign treatment based on propensity score
    t = np.random.binomial(1, propensity_score)
    
    # Generate baseline outcome (without treatment)
    baseline = 2 + X[:, 0] + X[:, 1] * X[:, 2] + np.random.normal(0, 1, size=n_samples)
    
    # Generate heterogeneous treatment effect
    if treatment_effect_coef is None:
        treatment_effect_coef = np.random.uniform(-1, 1, size=n_features)
    
    # True treatment effect varies by features
    true_effect = X.dot(treatment_effect_coef)
    
    # Generate observed outcome
    y = baseline + t * true_effect
    
    return X, t, y, true_effect


def evaluate_causal_model(model, X, t, y, true_effect=None):
    """
    Evaluate a causal model.
    
    Parameters:
    -----------
    model : object
        Fitted causal model with predict_effect method
    X : array-like
        Feature matrix
    t : array-like
        Treatment assignment
    y : array-like
        Outcome variable
    true_effect : array-like, optional
        True treatment effect (if available)
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Predict treatment effects
    pred_effect = model.predict_effect(X)
    
    # Calculate metrics
    metrics = {}
    
    # If true effects are available
    if true_effect is not None:
        # Mean squared error of treatment effect
        metrics['effect_mse'] = np.mean((pred_effect - true_effect) ** 2)
        # R-squared of treatment effect
        metrics['effect_r2'] = 1 - metrics['effect_mse'] / np.var(true_effect)
    
    # Calculate policy value (average outcome if we treat based on predicted positive effect)
    treat_recommended = pred_effect > 0
    # Policy value = E[Y | T=1, τ(X) > 0] * P(τ(X) > 0) + E[Y | T=0, τ(X) <= 0] * P(τ(X) <= 0)
    policy_value = np.mean(y[np.logical_and(t == 1, treat_recommended)]) * np.mean(treat_recommended) + \
                   np.mean(y[np.logical_and(t == 0, ~treat_recommended)]) * np.mean(~treat_recommended)
    metrics['policy_value'] = policy_value
    
    return metrics


def split_data(X, t, y, test_size=0.3, random_state=None):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    t : array-like
        Treatment assignment
    y : array-like
        Outcome variable
    test_size : float
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, t_train, t_test, y_train, y_test : array-like
        Split data
    """
    return train_test_split(X, t, y, test_size=test_size, random_state=random_state)
