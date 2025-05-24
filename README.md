# CHD_Prediction
Machine learning coronary heart disease prediction system


## Project Introduction  
The AI-CHD project builds a high-precision and interpretable **early prediction system for coronary heart disease** based on the Framingham Heart Study data from Kaggle, combining machine learning with SHAP interpretability algorithms. By integrating gradient boosting classifiers, random forests, XGBoost and other models, and optimizing data preprocessing and feature engineering, it achieves accurate prediction of coronary heart disease risk. An interactive web tool is developed to promote the transformation of coronary heart disease prevention and treatment from a "treatment center" to a "prevention center".


## Core Technical Architecture  
### 1. Data Processing and Feature Engineering  
- **Data Preprocessing**:  
  - Missing value imputation (mode filling), outlier removal (IQR method), duplicate value handling.  
  - Class imbalance solution: Adopting **SMOTEENN + SMOTETomek** two-stage resampling technology to balance the ratio of positive and negative samples.  
- **Feature Selection**:  
  - **Chi-square test + mutual information + F-test** two-layer screening method, comprehensively scoring and selecting the top 10 core features (such as systolic blood pressure, age, blood glucose, etc.).  
  - Feature standardization: MinMaxScaler normalization processing.  

### 2. Machine Learning Models  
- **Single Model Training**: Covers algorithms such as Logistic Regression, KNN, SVM, decision trees, and ensemble learning.  
- **Model Ensemble**:  
  - Adopting **Stacking ensemble technique**, using random forest, XGBoost, and gradient boosting tree as base models, and logistic regression as the meta-model.  
  - Internal validation performance: Accuracy 94.24%, AUC 0.9773, F1 score 0.9440.  
  - External validation: In 200 clinical samples from Nantong Affiliated Hospital, the accuracy reached 88.62%, verifying cross-cohort robustness.  

### 3. Interpretability Analysis (SHAP)  
- **Global Explanation**: Quantifying feature importance, revealing that age, systolic blood pressure, blood glucose, and total cholesterol are core risk factors.  
- **Single Sample Explanation**: Analyzing individual risk driving factors through SHAP force-directed graphs to assist doctors in formulating personalized intervention plans.  
- **Model Comparison**: Visualization of sensitivity differences of different base models to features (e.g., XGBoost is more sensitive to blood glucose, while random forest has weaker capture of hypertension history).  

### 4. Web Application Development  
- **Functions**: Input 10 physiological indicators (such as blood pressure, blood glucose, smoking amount, etc.), and real-time output risk prediction results, SHAP explanation diagrams, and personalized health suggestions.  


## Key Achievements  
| Indicator      | Integrated Model Performance |
|----------------|------------------------------|
| Accuracy       | 0.9424                       |
| AUC            | 0.9773                       |
| Sensitivity (Sn)| 0.9426                       |
| Specificity (Sp)| 0.9423                       |
| F1 Score       | 0.9440                       |

## Directory Description

- `data/`: Contains datasets used for training and testing the models.
- `analysis_models/`: Core code directory.
  - `models/`: "*.pkl"
  - `app.py`: Web application code for deploying the model.
  - `CHD_analysis.ipynb`: Jupyter notebook for data processing, model training, and integration.
- `README.md`: Project documentation file.

## Launch Web Application
```bash
streamlit run app.py
