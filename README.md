# A Multi-Paradigm Machine Learning Framework for Student Stress Prediction Across Heterogeneous Datasets

This repository contains the experimental code, datasets information, result files, figures, and supporting materials for the research paper:

**A Multi-Paradigm Machine Learning Framework for Student Stress Prediction Across Heterogeneous Datasets**

The study evaluates student stress prediction using multiple machine learning paradigms across three heterogeneous datasets. The experiments focus on feature representation, feature selection, supervised learning, ensemble learning, semi-supervised learning, unsupervised clustering, multimodal regression, modality ablation, and target-construction sensitivity analysis.

---

## Overview

Student stress prediction performance can vary considerably depending on the dataset structure, feature representation, target definition, validation strategy, and available modalities.

This work evaluates these factors using a dataset-aware framework across three different student stress datasets:

1. **Student Stress Factors Dataset**
   - 1,100 observations
   - 20 predictor features
   - 3 stress classes
   - Questionnaire-based psychological, academic, social, and behavioral factors

2. **SSAQS Dataset**
   - Longitudinal student stress and anxiety dataset
   - Questionnaire information combined with wearable measurements
   - Wearable modalities include HRV, oxygen, sleep, steps, and physical activity
   - Evaluated as a continuous stress regression problem

3. **University Student Stress Dataset**
   - 3,000 student observations
   - 18 attributes
   - Stress levels represented as Low, Medium, and High
   - Includes a calculated `Stress_Score`
   - Target-construction sensitivity analysis was performed

---

## Research Objectives

The main objectives of this work are:

- Compare different feature representations for student stress classification.
- Perform leakage-aware feature selection.
- Compare multiple supervised machine learning algorithms.
- Evaluate a heterogeneous ensemble of Random Forest, XGBoost, and SVM.
- Study classification performance under reduced label availability.
- Evaluate semi-supervised learning methods.
- Analyze student data using unsupervised clustering.
- Evaluate longitudinal multimodal stress regression using questionnaire and wearable data.
- Perform modality ablation analysis.
- Investigate the effect of target construction on classification performance.

---

## Machine Learning Methods

### Supervised Learning

The following classification models were evaluated on the primary dataset:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Extra Trees
- XGBoost

### Heterogeneous Ensemble

A heterogeneous ensemble was constructed using:

- Random Forest
- XGBoost
- Support Vector Machine

Performance-based weighting was used to combine the component models.

### Semi-Supervised Learning

The following methods were evaluated under different label-availability settings:

- Label Propagation
- Label Spreading
- Self-Training

Random Forest was also used as a supervised reference.

### Unsupervised Learning

The following clustering algorithms were evaluated:

- K-Means
- Agglomerative Clustering
- Gaussian Mixture Model

### Regression

The SSAQS dataset was evaluated using:

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost

---

## Feature Selection

For the Student Stress Factors dataset, the original 20-feature representation was compared with an expanded 24-feature representation.

Leakage-aware feature selection was then evaluated using:

- 20 features
- 15 features
- 10 features
- 8 features
- 5 features

The selected 10-feature representation achieved a mean five-fold validation accuracy of **88.73%**, compared with **87.45%** for the original 20-feature representation.

The selected features were:

1. `anxiety_level`
2. `self_esteem`
3. `depression`
4. `headache`
5. `blood_pressure`
6. `sleep_quality`
7. `teacher_student_relationship`
8. `future_career_concerns`
9. `extracurricular_activities`
10. `bullying`

---

## Main Results

### Primary Dataset

The heterogeneous ensemble achieved:

- **88.91%** mean five-fold validation accuracy
- **89.55%** accuracy on the independent test set
- **89.69%** precision
- **89.55%** recall
- **89.51%** F1-score

The independent test set contained:

- 880 training observations
- 220 testing observations

### Semi-Supervised and Unsupervised Analysis

At 50% label availability:

- Random Forest: **89.09%**
- Self-Training: **90.45%**

At 10% label availability:

- Random Forest: **89.09%**
- Self-Training: **89.09%**

For unsupervised clustering:

- K-Means Silhouette Score: **0.4655**
- K-Means ARI: **0.6487**
- Agglomerative Clustering NMI: **0.6718**

### SSAQS Regression

Gradient Boosting obtained:

- MAE: **13.98**
- RMSE: **18.55**
- R²: **0.419**

Modality-level experiments showed that adding wearable modalities did not consistently improve regression performance.

### Target-Construction Sensitivity

For the University Student Stress Dataset, standard classification produced very high accuracy, including:

- Logistic Regression: **99.87%**

After excluding variables associated with the available target-construction procedure:

- Logistic Regression: **53.53%**
- SVM: **53.17%**
- Random Forest: **50.53%**

This experiment highlights the importance of examining how derived targets are constructed when interpreting predictive performance.

---

## Repository Structure

```text
student-stress-prediction/
│
├── README.md
│
├── paper/
│   ├── conference_paper.pdf
│   └── conference_paper.docx
│
├── datasets/
│   ├── dataset1/
│   ├── dataset2/
│   └── dataset3/
│
├── preprocessing/
│   ├── feature_groups.py
│   └── leakage_safe_selection.py
│
├── supervised/
│   ├── step9_all_supervised_results.csv
│   ├── hybrid_ensemble.py
│   └── ...
│
├── semi_supervised/
│   ├── semi_supervised_all.py
│   ├── label_efficiency.py
│   └── ...
│
├── unsupervised/
│   ├── unsupervised_all.py
│   └── ...
│
├── regression/
│   ├── dataset2_24h_features.csv
│   ├── modality_ablation
│   └── ...
│
├── sensitivity_analysis/
│   └── ...
│
├── results/
│   ├── confusion_matrix
│   ├── result_tables
│   └── ...
│
└── figures/
    ├── framework.png
    └── confusion_matrix.png
