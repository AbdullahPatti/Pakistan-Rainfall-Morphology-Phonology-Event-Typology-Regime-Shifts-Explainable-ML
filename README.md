# Machine Learning Based Structural Characterization of Rainfall Event Typology and Regime Shifts in Pakistan (1981–2024)

This repository contains the complete implementation for an event-based machine learning framework that analyzes the **structural morphology of rainfall events in Pakistan** over a 44-year period.

Traditional rainfall studies rely on accumulated totals, which hide how rainfall actually unfolds in time. Two regions may receive identical rainfall totals but differ significantly in **onset speed, peak intensity, and recession behavior**. These structural properties determine flooding risk, crop stress, and groundwater recharge.

This project extracts rainfall events from dekadal precipitation records, constructs **morphological features describing event shape**, discovers natural rainfall typologies through clustering, detects **long-term regime shifts**, and evaluates machine learning models for typology classification.

The full research work is implemented here along with visualization and model evaluation.

---

# Project Overview

The pipeline processes **44 years of rainfall data (1981–2024)** covering **30 districts of Pakistan** and performs:

1. Rainfall event segmentation  
2. Morphological feature engineering  
3. Rainfall typology discovery using clustering  
4. Regime shift detection in long-term rainfall patterns  
5. Machine learning classification of rainfall typologies

Key statistics:

- **47,070 dekadal rainfall records**
- **30 districts**
- **4,563 extracted rainfall events**
- **7 rainfall typology classes discovered**

---

# Dataset

Dataset source:

WFP Rainfall Indicators Dataset (Pakistan)

The dataset contains **10-day rainfall observations** for multiple districts.

Main variables include:

| Feature | Description |
|------|------|
| date | Dekadal timestamp |
| adm2_id | District identifier |
| ADM2_PCODE | District administrative code |
| rfh | Rainfall total for dekad (mm) |
| rfh_avg | Long-term rainfall average |
| r1h | 1-month rolling rainfall |
| r3h | 3-month rolling rainfall |
| rfq | Dekadal rainfall quantile |
| r1q | 1-month rainfall quantile |
| r3q | 3-month rainfall quantile |

Total records: **47,070**

---

# Methodology Pipeline

## 1. Rainfall Event Segmentation

Dekadal rainfall records are converted into discrete **rainfall events**.

A rainfall event is defined as:

- Consecutive dekads with rainfall **greater than 5 mm**
- Event ends when rainfall falls below this threshold

Result:
47070 records → 4563 rainfall events


---

# 2. Morphological Feature Engineering

Each rainfall event is represented using **8 structural features** describing its temporal shape.

Features include:

| Feature | Description |
|------|------|
| Event Volume | Total rainfall accumulated during event |
| Mean Intensity | Average rainfall per dekad |
| Max Intensity | Maximum rainfall in a dekad |
| Duration | Number of dekads in event |
| Peak Anomaly | Difference from historical mean |
| Rise Gradient | Speed of rainfall increase |
| Decay Gradient | Speed of rainfall decrease |
| RCVR | Rainfall Consonant-Vowel Ratio (wet/dry ratio before event) |

These features encode **temporal rainfall structure**, not just magnitude.

---

# 3. Rainfall Typology Discovery

Rainfall events are clustered into structural types using:

**Gaussian Mixture Models (GMM)**

Reasons for choosing GMM:

- Allows **soft cluster membership**
- Handles **different cluster variances**
- Better suited for overlapping rainfall structures than k-means

Optimal number of clusters determined using:

- Bayesian Information Criterion (BIC)
- Silhouette score

Result:
Optimal clusters = 7 rainfall typology classes


---

# 4. Regime Shift Detection

To detect long-term structural changes in rainfall patterns, the pipeline applies:

**PELT (Pruned Exact Linear Time) changepoint detection**

Input:

Annual proportions of rainfall typology classes per district.

Outcome:

A persistent **regime shift around 1996** in rainfall event composition across several districts.

This indicates that rainfall structure changed even when total rainfall did not necessarily change.

---

# 5. Machine Learning Classification

The project evaluates whether rainfall typologies can be predicted from morphological features.

Models tested:

- Logistic Regression
- Random Forest
- XGBoost
- Artificial Neural Network (ANN)

Train/Test Split:
80% training
20% testing

Hyperparameter tuning:
5-fold cross validation


---

# Model Performance

| Model | Accuracy | Macro F1 |
|------|------|------|
| Logistic Regression | 0.80 | 0.76 |
| ANN | 0.954 | 0.91 |
| XGBoost | 0.963 | 0.93 |
| Random Forest | **0.966** | **0.94** |

Best model:

**Random Forest**

Important insight:

High overall accuracy hides minority class difficulty.  
Rare typologies have lower recall due to class imbalance.

---

# Feature Importance

Top features identified by Random Forest:

1. RCVR
2. Duration
3. Event Volume
4. Rise Gradient
5. Decay Gradient

This shows rainfall typology is driven more by **temporal structure** than raw intensity.

---


---

# Dashboard

Interactive rainfall analysis dashboard:

https://pakistan-rainfall-analysis.streamlit.app/

The dashboard allows:

- Typology distribution visualization
- Temporal rainfall pattern exploration
- Model prediction insights

---

# Key Findings

1. Rainfall events in Pakistan can be categorized into **7 structural typologies**.
2. Event morphology provides more insight than simple rainfall totals.
3. A **major structural rainfall shift occurred around 1996**.
4. Machine learning models can classify rainfall event types with **96% accuracy**.
5. Temporal features such as duration and antecedent moisture are more informative than intensity.

---

# Limitations

- Dekadal resolution hides sub-10-day rainfall bursts.
- Analysis performed at district level.
- Some rainfall types are underrepresented.
- Event gradients assume linear intensity progression.

---

# Future Work

Future improvements include:

- Using **daily rainfall data**
- Applying **SHAP explainability**
- Adding atmospheric variables
- Addressing class imbalance with **SMOTE**
- Improving minority typology prediction
- Integrating seasonal forecast models

---

# Author

Abdullah Haroon  
Computer Science Student  
FAST-NUCES Lahore

---

# License

This project is released for research and academic use.

