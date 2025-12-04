# Big Data Analytics Project: Obesity Level Estimation

## 1. Project Description
This project implements the complete Big Data Analytics lifecycle to predict obesity levels based on eating habits, physical activity, and demographic information. Using R programming, we apply multiple machine learning algorithms (Decision Trees, SVM, Naive Bayes) and clustering techniques (K-Means) to classify individuals into seven obesity categories and predict BMI values.

## 2. Dataset & Variables

**Source**: [UCI Machine Learning Repository - Obesity Levels Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

**Size**: 2,111 observations × 17 variables

### Variables

| Variable | Type | Description | Range/Values |
|----------|------|-------------|--------------|
| Gender | Categorical | Gender | Female/Male |
| Age | Numeric | Age in years | 14-61 |
| Height | Numeric | Height in meters | 1.45-1.98 |
| Weight | Numeric | Weight in kilograms | 39-173 |
| family_history_with_overweight | Categorical | Family history of overweight | yes/no |
| FAVC | Categorical | Frequent high caloric food consumption | yes/no |
| FCVC | Numeric | Vegetable consumption frequency | 1-3 |
| NCP | Numeric | Number of main meals daily | 1-4 |
| CAEC | Categorical | Food consumption between meals | Never/Sometimes/Frequently/Always |
| SMOKE | Categorical | Smoking habit | yes/no |
| CH2O | Numeric | Daily water consumption (liters) | 1-3 |
| SCC | Categorical | Calorie consumption monitoring | yes/no |
| FAF | Numeric | Physical activity frequency (days/week) | 0-3 |
| TUE | Numeric | Technology device usage (hours) | 0-2 |
| CALC | Categorical | Alcohol consumption | Never/Sometimes/Frequently |
| MTRANS | Categorical | Transportation method | Walking/Bike/Motorbike/Public_Transportation/Automobile |
| **NObeyesdad** | **Categorical** | **Target: Obesity level** | **7 categories** |

### Target Variable Categories
1. **Insufficient_Weight**: BMI < 18.5
2. **Normal_Weight**: 18.5 ≤ BMI < 25
3. **Overweight_Level_I**: 25 ≤ BMI < 27
4. **Overweight_Level_II**: 27 ≤ BMI < 30
5. **Obesity_Type_I**: 30 ≤ BMI < 35
6. **Obesity_Type_II**: 35 ≤ BMI < 40
7. **Obesity_Type_III**: BMI ≥ 40

## 3. Problem Definition & Objectives

### Problem Statement
Obesity is a critical global health issue. Understanding lifestyle and demographic factors that contribute to obesity can inform preventive strategies and personalized health interventions.

### Project Objectives
1. Identify significant predictors of obesity levels
2. Build and compare classification models (Decision Tree, SVM, Naive Bayes)
3. Discover natural groupings through K-Means clustering
4. Predict continuous BMI values using linear regression
5. Generate actionable health insights

## 4. Data Visualization & Analysis

### Key Visualizations

**1. Age Distribution (Histogram)**
- Most individuals aged 20-30 years
- Right-skewed distribution
- Younger population predominance

**2. BMI Distribution (Histogram)**
- Multimodal distribution with peaks at 25, 30, and 45
- Clear separation between obesity categories

**3. Weight & Height Distributions (Boxplots)**
- Weight: 39-173 kg, outliers above 140 kg
- Height: 1.45-1.98 m, normally distributed

**4. Obesity Levels Distribution (Bar Plot)**
- Obesity Type III most prevalent
- Balanced representation across all categories

**5. Gender Distribution (Bar Plot)**
- Males: 1,068 (50.6%)
- Females: 1,043 (49.4%)
- Gender-balanced dataset

**6. Physical Activity Frequency (Bar Plot)**
- Majority exercise 0-1 days/week
- Low physical activity overall

**7. Transportation Method (Bar Plot)**
- Public Transportation: 1,582 individuals (74.9%)
- Walking associated with lower average BMI

**8. Weight vs Height Scatter Plot (Colored by Obesity)**
- Clear clustering by obesity category
- Strong positive correlation between weight, height, and obesity

**9. BMI by Obesity Level (Boxplot)**
- Non-overlapping BMI ranges per category
- BMI is excellent predictor of obesity classification

**10. Family History Distribution (Bar Plot)**
- 1,726 with family history (81.8%)
- 385 without (18.2%)
- Strong genetic component evident

## 5. Data Preprocessing

### Methods Applied

**Missing Values**
- Check performed: No missing values found
- Backup strategy: Mean imputation for numeric columns

**Outlier Detection**
- Method: Boxplot visualization
- Action: Outliers retained (valid extreme values)

**Negative Value Correction**
- Method: Multiply by -1 if found
- Result: None detected

**Feature Engineering**
- Created BMI variable: `BMI = Weight / (Height²)`
- Rationale: Standard obesity measure

**Data Type Conversion**
- Converted all categorical variables to factors
- Ensures proper statistical analysis in R

## 6. Hypothesis Testing

### Hypothesis 1: Gender vs Obesity Levels
- **Null Hypothesis (H₀)**: Gender and obesity are independent
- **Test**: Chi-Square Test
- **Results**: χ² = 657.75, df = 6, p < 2.2e-16
- **Interpretation**: ✅ Reject H₀. Gender significantly affects obesity levels

### Hypothesis 2: Family History vs Obesity
- **Null Hypothesis (H₀)**: Family history and obesity are independent
- **Test**: Chi-Square Test
- **Results**: χ² = 621.98, df = 6, p < 2.2e-16
- **Interpretation**: ✅ Reject H₀. Family history is a very strong predictor

### Hypothesis 3: High Caloric Food Consumption vs Obesity
- **Null Hypothesis (H₀)**: High caloric food consumption and obesity are independent
- **Test**: Chi-Square Test
- **Results**: χ² = 233.34, df = 6, p < 2.2e-16
- **Interpretation**: ✅ Reject H₀. High caloric food significantly increases obesity risk

## 7. Machine Learning Dataset Preparation

### Train-Test Split
- **Training Set**: 1,485 observations (70%)
- **Testing Set**: 626 observations (30%)
- **Method**: Random sampling with `set.seed(123)` for reproducibility

### Features Used
All 16 predictor variables used for classification models. For BMI regression, only numeric predictors were used (Age, Weight, Height, FCVC, NCP, CH2O, FAF, TUE).

## 8. Analytical Techniques & Justification

### Technique 1: K-Means Clustering
- **Purpose**: Discover natural groupings in data
- **Configuration**: k=7 (matching obesity categories), nstart=25
- **Justification**: Unsupervised learning to validate if natural clusters align with medical obesity categories
- **Results**: 93.2% variance explained, clusters align well with actual categories

### Technique 2: Decision Tree (Conditional Inference Tree)
- **Algorithm**: `party::ctree()`
- **Purpose**: Classification with interpretability
- **Justification**: Handles non-linear relationships, provides visual decision rules
- **Configuration**: Automatic pruning, significance level 0.05
- **Results**: 90.89% accuracy, 46 terminal nodes

### Technique 3: Support Vector Machine (SVM)
- **Algorithm**: `e1071::svm()` with RBF kernel
- **Purpose**: High-accuracy classification
- **Justification**: Excellent for high-dimensional data, handles non-linear boundaries
- **Configuration**: Radial kernel, default hyperparameters
- **Results**: 92.01% accuracy (best model)

### Technique 4: Naive Bayes
- **Algorithm**: `e1071::naiveBayes()`
- **Purpose**: Probabilistic baseline classifier
- **Justification**: Fast, provides probability estimates, baseline comparison
- **Configuration**: Default Laplace smoothing
- **Results**: 66.93% accuracy

### Technique 5: Linear Regression
- **Algorithm**: `lm()`
- **Purpose**: Predict continuous BMI values
- **Justification**: Understand linear relationships between predictors and BMI
- **Results**: R² = 0.9897, correlation = 0.9949

## 9. Performance Measures & Evaluation

### Classification Models

| Rank | Algorithm | Accuracy | Strengths | Use Case |
|------|-----------|----------|-----------|----------|
| 🥇 1 | **SVM** | **92.01%** | Best accuracy, handles non-linearity | Production deployment |
| 🥈 2 | Decision Tree | 90.89% | Interpretable, visual rules | Clinical decision support |
| 🥉 3 | Naive Bayes | 66.93% | Fast, probabilistic | Baseline/rapid screening |

### Clustering Evaluation
- **Between-cluster SS / Total SS**: 93.2% (excellent separation)
- **Cluster sizes**: Well-distributed (136-423 per cluster)
- **Alignment**: Strong correspondence with actual obesity categories

### Regression Evaluation
- **R-squared**: 0.9897 (98.97% variance explained)
- **Correlation**: 0.9949 (near-perfect prediction)
- **Residual Standard Error**: 0.811
- **Significant Predictors**: Weight (strongest), Height, Age, FCVC, NCP, FAF (negative)

## 10. Project Findings & Quantification

### Finding 1: Model Performance
- **SVM achieved 92.01% accuracy**, outperforming other models
- Decision Tree (90.89%) provides best interpretability trade-off
- Naive Bayes (66.93%) unsuitable due to feature dependence

### Finding 2: Feature Importance
1. **Weight**: Primary predictor (strongest decision tree split)
2. **Height**: Secondary predictor
3. **Gender**: Significant effect (p < 2.2e-16)
4. **Family History**: 4.5× higher obesity risk
5. **Physical Activity (FAF)**: -0.099 BMI reduction per exercise day

### Finding 3: Transportation Impact
| Transportation | Avg BMI | Difference from Walking |
|----------------|---------|------------------------|
| Public Transportation | 30.10 | +6.44 |
| Automobile | 29.19 | +5.53 |
| Motorbike | 25.76 | +2.10 |
| Bike | 25.17 | +1.51 |
| **Walking** | **23.66** | **Baseline** |

**Insight**: Active transportation reduces BMI by ~6.5 points

### Finding 4: Gender Disparities
- Males: Higher Obesity Type II prevalence
- Females: Higher Obesity Type III prevalence (extreme obesity)
- Implication: Gender-specific interventions needed

### Finding 5: Lifestyle Correlations
- **Age vs Physical Activity**: r = -0.145 (older = less active)
- **Water Consumption vs BMI**: r = +0.144 (unexpected positive)
- **High Caloric Food**: 85% more likely to be obese

### Finding 6: Clustering Insights
- **Cluster 1** (BMI 44.59): Obesity Type III
- **Cluster 6** (BMI 18.39): Insufficient Weight
- **93.2% variance explained**: Strong natural groupings exist

### Finding 7: Predictive Power
- Linear regression R² = 0.9897 indicates BMI is **highly predictable** from lifestyle factors
- Weight and Height account for majority of variance
- Physical activity shows protective effect (negative coefficient)

---

## File Structure
```
Project/
├── Project.R                                      # Main R script
├── ObesityDataSet_raw_and_data_sinthetic.csv     # Dataset
├── README.md                                      # Documentation
└── Output/
    ├── obesity_cleaned.csv                        # Cleaned dataset
    └── Rplots.pdf                                 # Visualizations
```

## Running the Project
1. Open RStudio
2. Set working directory: `setwd("path/to/project")`
3. Run script: `source("Project.R")` or `Rscript Project.R`

---

**Project Team**: 
Mohamer Khaled – 21p0185

Mostafa Nasr – 21p0230

Ahmed Khaled – 21p0056

Ali Sherif – 21p0212

Saifeldin Mohamed – 21p0362

Mohamed Ehab Badr – 21p0375

**Date**: December 2025