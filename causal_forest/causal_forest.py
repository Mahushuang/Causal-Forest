"""
Implementation of a Causal Forest for estimating heterogeneous treatment effects.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from .causal_tree import CausalTree


class CausalForest:
    """
    A random forest for estimating heterogeneous treatment effects.
    
    This implementation uses the approach described in Athey and Imbens (2016)
    and Wager and Athey (2018) for causal forests.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=5, 
                 max_features='sqrt', bootstrap=True, n_jobs=None, random_state=None):
        """
        Initialize a causal forest.
        
        Parameters:
        -----------
        n_estimators : int, optional
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of the trees
        min_samples_leaf : int, optional
            Minimum number of samples required to be at a leaf node
        max_features : int, float, str, optional
            Number of features to consider when looking for the best split
        bootstrap : bool, optional
            Whether to bootstrap samples when building trees
        n_jobs : int, optional
            Number of jobs to run in parallel
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trees = []
        
    def _compute_max_features(self, n_features):
        """Compute the number of features to consider for splitting."""
        if isinstance(self.max_features, str):
            if self.max_features == 'auto' or self.max_features == 'sqrt':
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == 'log2':
                return max(1, int(np.log2(n_features)))
            else:
                raise ValueError("Invalid value for max_features. Allowed string values are 'auto', 'sqrt', or 'log2'.")
        elif self.max_features is None:
            return n_features
        elif isinstance(self.max_features, (int, float)):
            if isinstance(self.max_features, int):
                return self.max_features
            else:
                return max(1, int(self.max_features * n_features))
        else:
            raise ValueError("Invalid value for max_features. Allowed values are 'auto', 'sqrt', 'log2', int or float.")
    
    def _fit_tree(self, tree_idx, X, t, y, sample_weight, random_state):
        """Fit a single tree in the forest."""
        n_samples = X.shape[0]
        random_state = check_random_state(random_state)
        
        # Bootstrap samples if required
        if self.bootstrap:
            indices = random_state.randint(0, n_samples, n_samples)
            X_tree = X[indices]
            t_tree = t[indices]
            y_tree = y[indices]
            if sample_weight is not None:
                sample_weight_tree = sample_weight[indices]
            else:
                sample_weight_tree = None
        else:
            X_tree = X
            t_tree = t
            y_tree = y
            sample_weight_tree = sample_weight
        
        # Create and fit a tree
        tree = CausalTree(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state.randint(0, np.iinfo(np.int32).max)
        )
        tree.fit(X_tree, t_tree, y_tree, sample_weight=sample_weight_tree)
        
        return tree
        
    def fit(self, X, t, y, sample_weight=None):
        """
        Fit the causal forest.
        
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
        # Convert inputs to numpy arrays
        X = np.asarray(X)
        t = np.asarray(t)
        y = np.asarray(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        
        # Check dimensions
        n_samples, n_features = X.shape
        
        # Compute actual max_features
        self.max_features_ = self._compute_max_features(n_features)
        
        # Initialize random state
        random_state = check_random_state(self.random_state)
        
        # Fit trees in parallel
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_tree)(
                i, X, t, y, sample_weight, 
                random_state.randint(0, np.iinfo(np.int32).max)
            )
            for i in range(self.n_estimators)
        )
        
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
        if not self.trees:
            raise ValueError("Forest has not been fitted yet.")
        
        X = np.asarray(X)
        
        # Predict with each tree and average
        all_predictions = np.array([tree.predict_effect(X) for tree in self.trees])
        return np.mean(all_predictions, axis=0)
    
    def feature_importances(self):
        """
        Get feature importances from the forest.
        
        Returns:
        --------
        feature_importances : array-like of shape (n_features,)
            The feature importances based on the Gini importance.
        """
        if not self.trees:
            raise ValueError("Forest has not been fitted yet.")
        
        # Get feature importances from each tree and average
        all_importances = []
        for tree in self.trees:
            if hasattr(tree.tree, 'feature_importances_'):
                all_importances.append(tree.tree.feature_importances_)
        
        if not all_importances:
            raise ValueError("No feature importances available.")
        
        return np.mean(all_importances, axis=0)
