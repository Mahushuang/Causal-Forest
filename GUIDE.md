# 因果森林(Causal Forest)使用指南

作者: Mahushuang

## 简介

因果森林(Causal Forest)是一种强大的因果推断工具，能够:
- 通过双重机器学习框架，准确区分相关性与因果性
- 在控制大量混淆变量后，保持较高的预测精度和解释性
- 识别不同子群体中的异质性处理效应

本项目提供了因果森林算法的完整实现，包括示例代码和可视化工具。

## 快速开始

### 安装

1. 克隆仓库:
```
git clone https://github.com/Mahushuang/causal-forest.git
cd causal-forest
```

2. 安装依赖:
```
pip install -r requirements.txt
```

### 运行示例

1. 合成数据示例 (推荐新用户先尝试):
```
python examples/synthetic_data_example.py
```

2. 真实数据示例:
```
python examples/real_data_example.py
```

## 详细使用说明

### 数据准备

因果森林需要以下数据:
- 特征矩阵 `X`: 形状为 (n_samples, n_features) 的数组
- 处理变量 `t`: 形状为 (n_samples,) 的二元数组 (0或1)
- 结果变量 `y`: 形状为 (n_samples,) 的数组

```python
# 示例: 准备数据
import numpy as np
from sklearn.model_selection import train_test_split

# 加载或生成数据
X = ...  # 特征矩阵
t = ...  # 处理变量 (0或1)
y = ...  # 结果变量

# 分割训练集和测试集
X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
    X, t, y, test_size=0.3, random_state=42
)
```

### 模型训练

```python
from causal_forest import CausalForest

# 初始化模型
cf = CausalForest(
    n_estimators=100,  # 树的数量
    max_depth=5,       # 树的最大深度
    min_samples_leaf=10,  # 叶节点的最小样本数
    random_state=42    # 随机种子
)

# 训练模型
cf.fit(X_train, t_train, y_train)
```

### 预测处理效应

```python
# 预测处理效应
treatment_effects = cf.predict_effect(X_test)

# 计算平均处理效应
ate = np.mean(treatment_effects)
print(f"平均处理效应 (ATE): {ate:.2f}")
```

### 结果分析

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 创建包含特征和处理效应的数据框
results_df = pd.DataFrame(X_test, columns=[f"特征{i}" for i in range(X_test.shape[1])])
results_df['处理效应'] = treatment_effects

# 绘制处理效应分布
plt.figure(figsize=(10, 6))
sns.histplot(treatment_effects, kde=True)
plt.axvline(ate, color='red', linestyle='--', label=f'ATE: {ate:.2f}')
plt.title('处理效应分布')
plt.xlabel('处理效应')
plt.legend()
plt.savefig('treatment_effect_distribution.png')

# 识别高处理效应和低处理效应的子群体
high_effect = results_df[results_df['处理效应'] > np.percentile(treatment_effects, 75)]
low_effect = results_df[results_df['处理效应'] < np.percentile(treatment_effects, 25)]

print("\n子群体分析:")
print(f"高处理效应组 (> {np.percentile(treatment_effects, 75):.2f}):")
print(high_effect.mean())
print(f"\n低处理效应组 (< {np.percentile(treatment_effects, 25):.2f}):")
print(low_effect.mean())
```

## 参数调优

以下是一些关键参数的调优建议:

1. **n_estimators**: 树的数量，通常越多越好，但会增加计算成本。建议值: 100-500

2. **max_depth**: 树的最大深度，较小的值可以防止过拟合。建议值: 3-10

3. **min_samples_leaf**: 叶节点的最小样本数，较大的值可以防止过拟合。建议值: 5-100

4. **max_features**: 寻找最佳分割时考虑的特征数量。建议值: 'sqrt'或'log2'

## 结果复现

为了确保结果可复现:

1. 使用相同的随机种子 (例如 random_state=42)
2. 使用相同的模型参数
3. 对数据进行相同的预处理步骤
4. 保存生成的图表和指标进行比较

## 常见问题

1. **问题**: 模型预测的处理效应都接近于零。
   **解决方案**: 检查数据中是否存在足够的变异性，增加样本量，或调整模型参数。

2. **问题**: 运行时内存错误。
   **解决方案**: 减少n_estimators，或使用更小的数据集进行测试。

3. **问题**: 处理效应估计不稳定。
   **解决方案**: 增加min_samples_leaf，减小max_depth，或增加n_estimators。

## 联系方式

如有问题或建议，请联系Mahushuang。
