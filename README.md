# paper-1

# Supporting Data for "Evaluating the use of traffic data in predicting weather normalized NOx concentrations from a single site in Manchester city centre, UK"

This repository contains data and code used in the paper:

**Title**: "Evaluating the use of traffic data in predicting weather normalized NOx concentrations from a single site in Manchester city centre, UK"  
**Authors**: Yuchen Zeng, Congbo Song, Hao Zhang, David Topping 
**Published in**: journal, 2025

## Structure

- `data/raw/`: AURN_metadata.RData (ERA5) 原始数据（说明来源）
- `data/processed/`: traffic.csv (AURN) 清洗后的数据（说明处理方法）
- `scripts/`: test_wtv(H).py 分析代码
- `figures/`: 生成的图表

## How to Use

请确保安装以下依赖：
Please make sure the following dependencies are installed:

- Python 3.x
- pandas
- numpy
- matplotlib
- xgboost
- scikit-learn
- shap
- matplotlib

run：
```bash
python scripts/analyze.py
