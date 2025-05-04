# Causal Forest Implementation

**Author: Mahushuang**

This project implements the Causal Forest algorithm for estimating heterogeneous treatment effects. Causal Forests are an extension of Random Forests designed specifically for causal inference problems, capable of accurately distinguishing between correlation and causation.

## Features

- Implementation of Causal Trees and Causal Forests
- Utility functions for generating synthetic data and evaluating models
- Example scripts demonstrating usage with synthetic and real-world data
- Visualization of treatment effects and feature importances

## Background

Causal Forests are a machine learning method for estimating heterogeneous treatment effects. Unlike traditional predictive models that focus on correlations, Causal Forests are designed to identify causal relationships between treatments and outcomes.

Key characteristics of Causal Forests:
- They use a double machine learning framework to accurately distinguish between correlation and causation
- They can control for confounding variables while maintaining high prediction accuracy
- They provide interpretable results that help understand which factors influence treatment effects

## Installation

1. Clone this repository:
```
git clone https://github.com/Mahushuang/causal-forest.git
cd causal-forest
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage Guide

### Basic Usage

```python
from causal_forest import CausalForest

# Initialize the model
cf = CausalForest(n_estimators=100, max_depth=5, min_samples_leaf=10)

# Fit the model
cf.fit(X, t, y)

# Predict treatment effects
treatment_effects = cf.predict_effect(X_new)
```

### Running Examples

The project includes example scripts that demonstrate how to use the Causal Forest implementation:

1. Synthetic data example (recommended for first-time users):
```
python examples/synthetic_data_example.py
```
This will generate synthetic data with known treatment effects, train a Causal Forest model, and produce visualizations showing the model's performance.

2. Real-world data example:
```
python examples/real_data_example.py
```
This example uses the LaLonde dataset to demonstrate how to apply Causal Forest to real-world data.

### Reproducing Results

To reproduce the exact results shown in the paper/presentation:

1. Use the same random seed (42) as in the examples
2. Follow the parameter settings specified in the example scripts
3. For custom datasets, ensure proper preprocessing (standardization, handling missing values)
4. Save and compare the generated visualizations

## Implementation Details

This implementation is based on the approach described in:
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests.

The key components are:

1. **CausalTree**: A decision tree modified to estimate treatment effects rather than predict outcomes.

2. **CausalForest**: An ensemble of Causal Trees that uses bootstrap aggregation (bagging) to improve the stability and accuracy of estimates.

3. **Utility Functions**: Helper functions for data generation, model evaluation, and visualization.

## Applications

Causal Forests can be applied in various domains:

- **Marketing**: Identifying which customer segments respond best to specific promotions
- **Healthcare**: Determining which patients benefit most from certain treatments
- **Policy Analysis**: Evaluating the heterogeneous effects of policy interventions
- **E-commerce**: Optimizing product recommendations based on causal impact on conversion

## Citation

If you use this implementation in your research or project, please cite:

```
Mahushuang. (2023). Causal Forest Implementation. GitHub Repository.
https://github.com/Mahushuang/causal-forest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or collaboration opportunities, please contact Mahushuang.
