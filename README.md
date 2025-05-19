# CHD_Prediction
Machine learning coronary heart disease prediction system
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://www.python.org/)
[![ML Libraries](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange.svg)](https://scikit-learn.org/)


## 项目简介  
本项目基于Kaggle的Framingham心脏研究数据，结合机器学习与SHAP可解释性算法，构建了一个高精度、可解释的**冠心病早期预测系统**。通过集成梯度提升分类器、随机森林、XGBoost等模型，结合数据预处理与特征工程优化，实现了对冠心病风险的精准预测，并开发了交互式网页工具，推动冠心病防治从“治疗中心”向“预防中心”转型。


## 核心技术架构  
### 1. 数据处理与特征工程  
- **数据预处理**：  
  - 缺失值填补（众数填充）、异常值剔除（IQR法）、重复值处理。  
  - 类别不平衡解决：采用**SMOTEENN + SMOTETomek**两阶段重采样技术，平衡正负样本比例。  
- **特征筛选**：  
  - 创新采用**卡方检验 + 互信息 + F检验**双层筛选方法，综合评分选取前10项核心特征（如收缩压、年龄、血糖等）。  
  - 特征标准化：MinMaxScaler归一化处理。  

### 2. 机器学习模型  
- **单模型训练**：涵盖Logistic Regression、KNN、SVM、决策树、集成学习等10种算法。  
- **模型集成**：  
  - 采用**Stacking堆叠集成技术**，以随机森林、XGBoost、梯度提升树为基模型，逻辑回归为元模型。  
  - 内部验证性能：准确率93.40%，AUC 0.9775，F1分数0.9350。  
  - 外部验证：在南通附属医院200例临床样本中，准确率达88.62%，验证跨队列鲁棒性。  

### 3. 可解释性分析（SHAP）  
- **全局解释**：量化特征重要性，揭示年龄、收缩压、血糖、总胆固醇为核心风险因子。  
- **单样本解释**：通过SHAP力导向图解析个体风险驱动因素，辅助医生制定个性化干预方案。  
- **模型对比**：不同基模型对特征的敏感性差异可视化（如XGBoost对血糖更敏感，随机森林对高血压病史捕捉较弱）。  

### 4. 网页应用开发  
- **功能**：输入10项生理指标（如血压、血糖、吸烟量等），实时输出风险预测结果、SHAP解释图及个性化健康建议。  
- **技术栈**：Flask框架 + HTML/CSS + ECharts可视化，支持本地化部署。  


## 关键成果  
| 指标         | 集成模型表现       |
|--------------|--------------------|
| 准确率       | 0.9340            |
| AUC          | 0.9775            |
| 敏感度（Sn） | 0.9395            |
| 特异度（Sp） | 0.9282            |
| F1分数       | 0.9350            |

- **临床价值**：通过特征交互效应分析（如血糖与总胆固醇协同升高加剧风险），为高危人群分层提供量化依据。  
- **工具开源**：开发免费易用的冠心病预测网页，降低基层医疗筛查门槛。  


## 项目结构  
project/
├─ data/ # 原始数据及预处理后数据集
├─ models/ # 训练好的模型文件（.pkl）
├─ src/ # 核心代码
│ ├─ data_preprocessing.py # 数据清洗与特征工程
│ ├─ model_training.py # 模型训练与集成
│ ├─ shap_explanation.py # SHAP 解释分析
│ └─ web_app/ # 网页应用代码
└─ requirements.txt # 依赖库列表

## 启动网页应用
streamlit run app.py

