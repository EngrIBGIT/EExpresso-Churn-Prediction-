# Expresso-Churn-Prediction-
Customer churn prediction using Expresso dataset: EDA, preprocessing, and model development

## Introduction
This project is focused on predicting customer churn using a dataset containing various features about customers. The goal is to build a model that can predict whether a customer will churn based on these features.

## Dataset Description
The dataset includes 19 variables:

- 15 Numeric Variables: Represent continuous data (e.g., top-up amount, frequency of recharge, monthly revenue, etc.)
- 4 Categorical Variables: Represent categorical data (e.g., region, tenure, top pack used, etc.)
- Target Variable: CHURN indicates whether the customer will churn (1 for churn, 0 for no churn).

## Key Steps
1.  Importing Libraries:

- All necessary Libraries; 
  - import pandas as pd
  - import numpy as np
  - import seaborn as sns
  - import matplotlib.pyplot as plt

  - from catboost import CatBoostClassifier
  - from sklearn.model_selection import train_test_split
  - from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix 

    are imported for data manipulation, visualization, and model building.


https://drive.google.com/file/d/1GmELQx3ZiIaBf0Xm7a1OzAr-it8W_r-W/view?usp=sharing
