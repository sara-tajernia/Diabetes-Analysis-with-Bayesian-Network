# A Bayesian Network for Modeling and Explaining Diabaese Prediction

## Table of Contents

- [A Bayesian Network for Modeling and Explaining Diabaese Prediction](#a-bayesian-network-for-modeling-and-explaining-diabaese-prediction)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Dataset Overview](#dataset-overview)
  - [EDA](#eda)
  - [Training Bayesian Network](#training-bayesian-network)
  - [Total Results Comparison](#total-results-comparison)
    - [Initial Observation: Basic Models](#initial-observation-basic-models)
    - [PreProcessing Methods Comparison](#preprocessing-methods-comparison)
      - [1. Mean Imputation + Quantile Discretization](#1-mean-imputation--quantile-discretization)
      - [2. Mean Imputation + KMeans Discretization](#2-mean-imputation--kmeans-discretization)
      - [3. KNN Imputation + Quantile Discretization](#3-knn-imputation--quantile-discretization)
      - [4. KNN Imputation + KMeans Discretization](#4-knn-imputation--kmeans-discretization)
    - [Model Structure Comparison (Non-Basic Models)](#model-structure-comparison-non-basic-models)
    - [Summary of Key Findings](#summary-of-key-findings)

## Abstract

This project presents a Bayesian Network designed to model and explain the prediction of diabetes based on various health and lifestyle factors. The network incorporates variables such as age, body mass index (BMI), physical activity, family history, and blood glucose levels to provide a probabilistic framework for understanding the relationships between these factors and the likelihood of developing diabetes. By leveraging the principles of Bayesian inference, the model aims to offer insights into the contributing factors of diabetes, enabling healthcare professionals to make informed decisions regarding prevention and treatment strategies. The network also facilitates explainability by allowing users to visualize how changes in specific variables impact the overall risk of diabetes, thereby enhancing interpretability and trust in the predictions made by the model.

## Dataset Overview

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The dataset contains the following attributes:

| Attribute                     | Description                                      |
|-------------------------------|--------------------------------------------------|
| Pregnancies                   | Number of times pregnant                         |
| Glucose                       | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| BloodPressure                 | Diastolic blood pressure (mm Hg)                 |
| SkinThickness                 | Triceps skin fold thickness (mm)                 |
| Insulin                       | 2-Hour serum insulin (mu U/ml)                   |
| BMI                           | Body mass index (weight in kg/(height in m)^2).  |
| DiabetesPedigreeFunction      | Diabetes pedigree function                       |
| Age                           | Age (years)                                      |
| Outcome                       | Class variable (0 or 1) indicating if the patient has diabetes (1) or not (0) |

The dataset consists of 768 instances, with 500 instances labeled as non-diabetic (0) and 268 instances labeled as diabetic (1). The features include both continuous and categorical variables, which will be used to construct the Bayesian Network for predicting diabetes.

## EDA

Exploratory Data Analysis (EDA) is a crucial step in the data analysis process. It helps us to better understand the given data, so that we can make sense out of it. If EDA is not done properly, it can hamper the further steps in the machine learning model building process. On the other hand, if done well, it may improve the efficacy of everything we do next. In order to perform EDA, we need to follow a systematic approach that involves several techniques. The following are some of the key steps involved in EDA:

1. Data Sourcing: This is the very first step of EDA, where we access data and load it into our system.

2. Data Cleaning: Once we have the data, we need to clean it by removing any inconsistencies, missing values, or outliers.

3. Analysing Dataset Characteristics: This step involves understanding the structure and properties of the dataset, such as data types, distributions, and relationships between variables.
In here We Analyse the dataset before and after cleaning it. So we can have a better understanding of cleaning process effects on the dataset.

On Data Analysis we do the following steps:

1. statistical summary of the dataset
2. plotting distributions of features
3. checking for missing values
4. checking for zero values in features where it is not possible
5. scatter plot of each feature vs Outcome
6. pair plot of features
7. heatmap of correlation matrix

## Training Bayesian Network

for Training Bayesian Network we will use pgmpy library. We will follow these steps:

1. Define the structure of the Bayesian Network which there is three options here:
   - we can define the structure manually based on domain knowledge.
   - we can use structure learning algorithms to learn the structure from data.
   - we can use a hybrid approach where we define some parts of the structure manually and learn other parts from data.
we will go through all three options and compare the results.

2. Find the Conditional Probability Distributions (CPDs) for each node in the network.
3. Perform inference on the Bayesian Network to make analyses with variable elimination. We will perform infrences to analyses the result of different scenarios.

## Total Results Comparison

In this study, we evaluated the impact of different preprocessing strategies and Bayesian Network structure learning methods on diabetes risk inference using the Indian Diabetes dataset.

Four preprocessing pipelines were considered:

- Mean Imputation + Quantile Discretization
- Mean Imputation + KMeans Discretization
- KNN Imputation + Quantile Discretization
- KNN Imputation + KMeans Discretization

For each preprocessing pipeline, five Bayesian Network structures were learned:

- Basic (fixed structure)
- Grouped
- Tree-based
- Hill Climbing with K2 score
- Hill Climbing with AIC score

This results in a total of **20 distinct models**, each evaluated across multiple clinically motivated inference questions.

---

### Initial Observation: Basic Models

Across all three glucose-related questions (Figures 1–3), the **basic models consistently produced identical probability distributions** regardless of the evidence provided. In all scenarios, the posterior probabilities of `Outcome=0` and `Outcome=1` remained approximately equal (≈ 0.5).

This behavior indicates that:

- The basic structures fail to encode meaningful conditional dependencies.
- Evidence provided to the model does not propagate effectively to the outcome node.

As a result, **basic models were excluded from further comparative analysis**, as they do not provide informative or discriminative inferences.

---

### PreProcessing Methods Comparison

After excluding the basic models, we compare the remaining preprocessing strategies based on the **average behavior across the three glucose-related questions**.

#### 1. Mean Imputation + Quantile Discretization

Models trained on mean-imputed and quantile-discretized data showed:

- Moderate sensitivity to changes in Age, BMI, and Glucose.
- Consistent trends across Grouped and Hill Climbing models.
- Relatively balanced uncertainty, as reflected by wider confidence intervals in several scenarios.

While this preprocessing strategy preserves global distributional properties, it appears to **smooth out extreme effects**, especially in high-risk glucose bins.

---

#### 2. Mean Imputation + KMeans Discretization

This preprocessing pipeline produced **sharper probability contrasts**, particularly in:

- Tree-based models
- Hill Climbing models with both K2 and AIC scores

In multiple scenarios, these models assigned **very high or very low diabetes probabilities**, suggesting that KMeans discretization captures localized structure in the feature space more effectively than quantile binning when paired with mean imputation.

However, some models exhibited overly confident predictions, which may indicate **overfitting to discretization-induced clusters**.

---

#### 3. KNN Imputation + Quantile Discretization

KNN-imputed datasets combined with quantile discretization demonstrated:

- More stable and consistent inference results across models
- Clear monotonic trends when Age or BMI increased under high Glucose conditions
- Reduced variance compared to mean-imputed counterparts

This suggests that KNN imputation better preserves multivariate relationships prior to discretization, allowing the Bayesian Networks to capture more realistic conditional dependencies.

---

#### 4. KNN Imputation + KMeans Discretization

Among all preprocessing strategies, **KNN Imputation + KMeans Discretization consistently produced the most interpretable and clinically plausible results**:

- High glucose combined with increasing age led to a clear increase in diabetes probability.
- High BMI amplified diabetes risk in younger individuals more distinctly than in other pipelines.
- Low BMI scenarios showed a noticeable protective effect even under high glucose conditions.

Hill Climbing models trained under this preprocessing pipeline demonstrated both **strong sensitivity to evidence** and **reasonable uncertainty**, making them particularly suitable for scenario-based risk analysis.

---

### Model Structure Comparison (Non-Basic Models)

Across all preprocessing pipelines, the following general trends were observed:

- **Grouped models** produced conservative and stable estimates, often serving as a baseline among non-basic structures.
- **Tree-based models** were highly sensitive to evidence but occasionally produced extreme probabilities.
- **Hill Climbing models (K2 and AIC)** provided the most expressive structures, with K2 favoring stronger dependencies and AIC yielding slightly more regularized graphs.

Notably, Hill Climbing models benefited the most from KNN-based imputation, indicating that structure learning is highly sensitive to the quality of missing-value handling.

---

### Summary of Key Findings

- Basic Bayesian Network structures are insufficient for meaningful inference and should be avoided.
- Preprocessing choice has a **substantial impact** on downstream probabilistic inference.
- KNN imputation consistently outperforms mean imputation in preserving conditional relationships.
- KMeans discretization enhances model sensitivity but should be paired with robust imputation.
- Hill Climbing–based Bayesian Networks, particularly with KNN + KMeans preprocessing, provide the most reliable and interpretable results.

Overall, the results highlight that **Bayesian Network inference quality is driven more by preprocessing decisions than by the choice of scoring function alone**, underscoring the importance of careful data preparation in probabilistic modeling.
