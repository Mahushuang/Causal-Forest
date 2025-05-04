"""
Example of using Causal Forest with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.inspection import permutation_importance

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_forest import CausalForest
from causal_forest.utils import generate_synthetic_data, evaluate_causal_model, split_data


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 2000
    n_features = 10
    
    # Create treatment effect that depends on first 3 features
    treatment_effect_coef = np.zeros(n_features)
    treatment_effect_coef[:3] = [1.0, -0.5, 0.8]
    
    X, t, y, true_effect = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        treatment_effect_coef=treatment_effect_coef,
        random_state=42
    )

    # Split data into training and testing sets
    X_train, X_test, t_train, t_test, y_train, y_test = split_data(
        X, t, y, test_size=0.3, random_state=42
    )

    # Get true effects for test set
    true_effect_test = X_test.dot(treatment_effect_coef)
    
    # Train a causal forest
    print("Training causal forest...")
    cf = CausalForest(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    cf.fit(X_train, t_train, y_train)
    
    # Predict treatment effects
    print("Predicting treatment effects...")
    pred_effect = cf.predict_effect(X_test)
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_causal_model(cf, X_test, t_test, y_test, true_effect_test)
    print(f"Evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Calculate feature importances
    print("Calculating feature importances...")
    try:
        importances = cf.feature_importances()
        feature_names = [f"Feature {i}" for i in range(n_features)]
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        plt.title('Feature Importances')
        plt.bar(range(n_features), importances[indices], align='center')
        plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        print("Feature importances saved to 'feature_importances.png'")
    except Exception as e:
        print(f"Could not calculate feature importances: {e}")
    
    # Plot true vs predicted treatment effects
    plt.figure(figsize=(10, 6))
    plt.scatter(true_effect_test, pred_effect, alpha=0.5)
    plt.plot([min(true_effect_test), max(true_effect_test)], 
             [min(true_effect_test), max(true_effect_test)], 
             'r--')
    plt.xlabel('True Treatment Effect')
    plt.ylabel('Predicted Treatment Effect')
    plt.title('True vs Predicted Treatment Effects')
    plt.tight_layout()
    plt.savefig('treatment_effects.png')
    print("Treatment effects plot saved to 'treatment_effects.png'")
    
    # Plot treatment effect heterogeneity
    plt.figure(figsize=(12, 8))
    
    # Create a dataframe for easier plotting
    df = pd.DataFrame({
        'Feature 0': X_test[:, 0],
        'Feature 1': X_test[:, 1],
        'Feature 2': X_test[:, 2],
        'True Effect': true_effect_test,
        'Predicted Effect': pred_effect
    })
    
    # Plot treatment effect by Feature 0
    plt.subplot(2, 2, 1)
    sns.regplot(x='Feature 0', y='True Effect', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    sns.regplot(x='Feature 0', y='Predicted Effect', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'blue'})
    plt.title('Treatment Effect by Feature 0')
    
    # Plot treatment effect by Feature 1
    plt.subplot(2, 2, 2)
    sns.regplot(x='Feature 1', y='True Effect', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    sns.regplot(x='Feature 1', y='Predicted Effect', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'blue'})
    plt.title('Treatment Effect by Feature 1')
    
    # Plot treatment effect by Feature 2
    plt.subplot(2, 2, 3)
    sns.regplot(x='Feature 2', y='True Effect', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    sns.regplot(x='Feature 2', y='Predicted Effect', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'blue'})
    plt.title('Treatment Effect by Feature 2')
    
    # Plot distribution of treatment effects
    plt.subplot(2, 2, 4)
    sns.kdeplot(true_effect_test, label='True Effect', fill=True, alpha=0.3)
    sns.kdeplot(pred_effect, label='Predicted Effect', fill=True, alpha=0.3)
    plt.title('Distribution of Treatment Effects')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('treatment_effect_heterogeneity.png')
    print("Treatment effect heterogeneity plot saved to 'treatment_effect_heterogeneity.png'")


if __name__ == "__main__":
    main()
