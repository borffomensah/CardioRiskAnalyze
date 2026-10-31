# Cardiovascular Risk Assessment Using Machine Learning: A Support Vector Approach

📊 **Project Overview**  
This project presents an interpretable machine learning framework designed to classify cardiovascular disease risk using clinical patient data. By analyzing specific physiological signatures such as cholesterol levels, resting blood pressure, and ST-segment configurations—this model provides a scalable strategy for early risk assessment and screening.  

🔗 **Live Demo:** https://heartdiseasesdetection2024.streamlit.app/

## Clinical Context
Cardiovascular diseases (CVDs) are the leading cause of global mortality. Traditional diagnostic scoring systems heavily depend on rigid, linear boundaries that often fail to adapt to complex, non-linear feature sets within diverse patient distributions. This study utilizes 14 key clinical features from the UCI Heart Disease dataset to map predictive attributes:

*   **Serum Cholesterol (`chol`):** Crucial metric for identifying lipid accumulation and plaque build-up.
*   **Resting Blood Pressure (`trestbps`):** Key baseline hemodynamic indicator monitored upon admission.
*   **Maximum Heart Rate (`thalach`):** Tracks physical performance thresholds, showcasing inverse proportion trends to positive cardiovascular targets.
*   **ST Depression (`oldpeak` & `slope`):** Direct electrocardiographic markers capturing exercise-induced ischemia relative to rest states.

## Data Acquisition & Properties
The dataset evaluates 270 structural clinical samples with balanced target profiles (55.6% no disease vs. 44.4% positive disease presence). 
*   **Null-Field Verification:** 0 missing values across all 14 baseline rows.
*   **Attribute Distribution:** Continuous features (`age`, `chol`, `thalach`, `trestbps`) alongside multi-class categorical arrays (`cp`, `restecg`, `slope`, `thal`).

## Machine Learning Pipeline
*   **Data Preprocessing:** Categorical features mapped via indexing encoders, and numerical kontinuier parameters normalized using continuous feature scaling to prevent hyperplane boundary skewing.
*   **Model Selection:** Support Vector Machine (SVM) Classifier, using a Radial Basis Function (RBF) kernel to map high-dimensional structural dependencies cleanly.
*   **Optimization:** Configured using an 80:20 stratified training-test split with comprehensive Grid Search tuning over regularization parameters ($C$) and gamma ($\gamma$) coefficients.

### Evaluation Metrics
The optimized binary model asset (`best_svm_model.pkl`) achieved stable test evaluation bounds:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 89.5% |
| **Precision** | 87.2% |
| **Recall** | 85.6% |
| **F1-Score** | 86.4% |
| **ROC-AUC** | 0.93 |

---
*Disclaimer: This project is for informational purposes only. For medical advice or diagnosis, consult a healthcare professional.*
