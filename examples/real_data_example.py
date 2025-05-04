"""
Example of using Causal Forest with a real-world dataset.

This example uses the LaLonde dataset, which is a commonly used dataset
for causal inference. It contains data from a job training program.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_forest import CausalForest


def load_lalonde_data():
    """
    Load the LaLonde dataset.
    
    If the dataset is not available locally, it will be downloaded from the web.
    
    Returns:
    --------
    X : array-like
        Feature matrix
    t : array-like
        Treatment assignment
    y : array-like
        Outcome variable
    """
    # Try to load from local file
    try:
        df = pd.read_csv('lalonde.csv')
    except FileNotFoundError:
        # Download from web if not available locally
        url = "https://raw.githubusercontent.com/mdcattaneo/replication-CCJM_2022_RESTAT/main/lalonde.csv"
        df = pd.read_csv(url)
        # Save locally for future use
        df.to_csv('lalonde.csv', index=False)
    
    # Extract treatment, outcome, and features
    t = df['treat'].values
    y = df['re78'].values  # 1978 earnings (outcome)
    
    # Features: age, education, race, married, earnings in 1974 and 1975
    features = ['age', 'educ', 'black', 'hisp', 'married', 're74', 're75']
    X = df[features].values
    
    return X, t, y, features


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Loading LaLonde dataset...")
    try:
        X, t, y, feature_names = load_lalonde_data()
    except Exception as e:
        print(f"Error loading LaLonde dataset: {e}")
        print("Using synthetic data instead...")
        
        # Generate synthetic data as fallback
        from causal_forest.utils import generate_synthetic_data
        n_samples = 2000
        n_features = 7
        X, t, y, _ = generate_synthetic_data(n_samples=n_samples, n_features=n_features, random_state=42)
        feature_names = [f"Feature {i}" for i in range(n_features)]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X_scaled, t, y, test_size=0.3, random_state=42
    )
    
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
    
    # Calculate average treatment effect
    ate = np.mean(pred_effect)
    print(f"Average Treatment Effect (ATE): {ate:.2f}")
    
    # Calculate feature importances
    print("Calculating feature importances...")
    try:
        importances = cf.feature_importances()
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        plt.title('Feature Importances')
        plt.bar(range(len(feature_names)), importances[indices], align='center')
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('real_data_feature_importances.png')
        print("Feature importances saved to 'real_data_feature_importances.png'")
    except Exception as e:
        print(f"Could not calculate feature importances: {e}")
    
    # Create a dataframe for the test data
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_df['Treatment Effect'] = pred_effect
    
    # Plot treatment effect heterogeneity
    plt.figure(figsize=(12, 10))
    
    # Plot treatment effect by each feature
    for i, feature in enumerate(feature_names):
        plt.subplot(3, 3, i+1)
        sns.regplot(x=feature, y='Treatment Effect', data=X_test_df, scatter_kws={'alpha': 0.3})
        plt.title(f'Treatment Effect by {feature}')
    
    plt.tight_layout()
    plt.savefig('real_data_treatment_effect_heterogeneity.png')
    print("Treatment effect heterogeneity plot saved to 'real_data_treatment_effect_heterogeneity.png'")
    
    # Plot distribution of treatment effects
    plt.figure(figsize=(10, 6))
    sns.histplot(pred_effect, kde=True)
    plt.axvline(ate, color='red', linestyle='--', label=f'ATE: {ate:.2f}')
    plt.title('Distribution of Treatment Effects')
    plt.xlabel('Treatment Effect')
    plt.legend()
    plt.tight_layout()
    plt.savefig('real_data_treatment_effect_distribution.png')
    print("Treatment effect distribution plot saved to 'real_data_treatment_effect_distribution.png'")
    
    # Identify subgroups with high and low treatment effects
    high_effect = X_test_df[X_test_df['Treatment Effect'] > np.percentile(pred_effect, 75)]
    low_effect = X_test_df[X_test_df['Treatment Effect'] < np.percentile(pred_effect, 25)]
    
    print("\nSubgroup Analysis:")
    print(f"High Treatment Effect Group (> {np.percentile(pred_effect, 75):.2f}):")
    print(high_effect.mean())
    print(f"\nLow Treatment Effect Group (< {np.percentile(pred_effect, 25):.2f}):")
    print(low_effect.mean())


if __name__ == "__main__":
    main()
