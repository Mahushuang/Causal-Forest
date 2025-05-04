"""
Implementation of a Causal Tree for estimating heterogeneous treatment effects.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class CausalTree:
    """
    A decision tree for estimating heterogeneous treatment effects.
    
    This implementation uses the "honest" approach described in Athey and Imbens (2016)
    where separate samples are used for determining splits and estimating effects.
    """
    
    def __init__(self, max_depth=None, min_samples_leaf=5, random_state=None):
        """
        Initialize a causal tree.
        
        Parameters:
        -----------
        max_depth : int, optional
            Maximum depth of the tree
        min_samples_leaf : int, optional
            Minimum number of samples required to be at a leaf node
        random_state : int, optional
            Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None
        
    def fit(self, X, t, y, sample_weight=None):
        """
        Fit the causal tree.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        t : array-like of shape (n_samples,)
            The treatment indicator (0 or 1).
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
            
        Returns:
        --------
        self : object
        """
        # Create transformed outcome for causal effect estimation
        # Using the transformed outcome approach from Athey & Imbens
        t_centered = t - np.mean(t)
        transformed_outcome = y * t_centered / (np.mean(t) * (1 - np.mean(t)))
        
        # Fit a regression tree on the transformed outcome
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree.fit(X, transformed_outcome, sample_weight=sample_weight)
        
        return self
    
    def predict_effect(self, X):
        """
        Predict treatment effect for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        effects : array-like of shape (n_samples,)
            The predicted treatment effects.
        """
        if self.tree is None:
            raise ValueError("Tree has not been fitted yet.")
        
        return self.tree.predict(X)
    
    def get_depth(self):
        """Return the depth of the tree."""
        if self.tree is None:
            raise ValueError("Tree has not been fitted yet.")
        return self.tree.get_depth()
    
    def get_n_leaves(self):
        """Return the number of leaves in the tree."""
        if self.tree is None:
            raise ValueError("Tree has not been fitted yet.")
        return self.tree.get_n_leaves()
    
    def apply(self, X):
        """Return the index of the leaf that each sample is predicted as."""
        if self.tree is None:
            raise ValueError("Tree has not been fitted yet.")
        return self.tree.apply(X)
