# Expresso-Churn-Prediction-
Customer churn prediction using Expresso dataset: EDA, preprocessing, and model development

  `NOTE: This structured approach is a guide to help one navigate through the data science pipeline of the expresso dataset, from understanding the data to building a predictive model.`
  
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
  - `import pandas as pd`
  - `import numpy as np`
  - `import seaborn as sns`
  - `import matplotlib.pyplot as plt`

  - `from catboost import CatBoostClassifier`
  - `from sklearn.model_selection import train_test_split`
  - `from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix` 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

    are imported for data manipulation, visualization, and model building.
    
2. Load the Data:
- Load the training, test, and sample submission datasets
  - `train = pd.read_csv("train.csv")`
  - `test = pd.read_csv("Test.csv")`
  - `sample_sub = pd.read_csv("sample_submission.csv")`

3.  Exploratory Data Analysis (EDA):

- Correlation Matrix: Helps to understand the relationships between numerical features.
- `Histograms:` Provide the distribution of numerical features.
- `Box Plots:` Show the relationship between numerical features and the target variable CHURN.
- `Violin Plots:` Visualize the distribution of the numerical features split by the target variable.

      train_num = train.select_dtypes(include=['number'])
      corr = train_num.corr()
      plt.figure(figsize=(12, 8))
      sns.heatmap(corr, annot=True, cmap='coolwarm')
      plt.title('Correlation Matrix')
      plt.show()`
    
      train_num_feat = ["FREQ_TOP_PACK", "REGULARITY", "ORANGE", "ON_NET", "DATA_VOLUME", "FREQUENCE", "REVENUE", "MONTANT"]
      train[train_num_feat].hist(figsize=(15, 15), bins=15)
      plt.suptitle('Distribution of Numerical Features')
      plt.show()
    
        for col in train_num_feat:
          plt.figure(figsize=(10, 5))
          sns.boxplot(x='CHURN', y=col, data=train)
          plt.title(f'Box Plot of {col} by CHURN')
          plt.show()
      
        for col in train_num_feat:
          plt.figure(figsize=(10, 5))
          sns.violinplot(x='CHURN', y=col, data=train)
          plt.title(f'Relationship between {col} and CHURN')
          plt.show()
  
4.  Correlation with Target Variable:

-  The correlation between each numerical feature and the target variable `CHURN` is calculated to understand which features might be important for prediction.

        corr_with_target = train_num.corr()['CHURN'].sort_values(ascending=False)
        print(corr_with_target)


corr_with_target = train_num.corr()['CHURN'].sort_values(ascending=False)
print(corr_with_target)


5. Visualize the Target Variable:

-  The distribution of the target variable `CHURN` is visualized to understand the balance of the classes (churn vs. no churn).

        sns.countplot(x='CHURN', data=train)
        plt.title('Distribution of Churn')
        plt.show()

6. Data Preprocessing:

- `Handling Missing Values:` Since the dataset has missing values, handling them properly (e.g., imputation) is essential.
- `Encoding Categorical Variables:` Convert categorical variables into numerical formats.
- `Feature Scaling:` Standardize or normalize the numerical features if necessary.

        churn = train['CHURN']
        train = train.drop('CHURN', axis=1)
        d_tt = pd.concat([train, test], sort=False)
7.  Handle Missing Values:

        # Calculate the percentage of missing values
        percentage_missing = (missing_values / len(d_tt)) * 100
        
        # Combine the missing values and their percentages into a DataFrame
        missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': percentage_missing})
        
        # Display the DataFrame sorted by percentage of missing values in descending order
        missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percentage', ascending=False)
        missing_data
8.  Handle Categorical Values:
   
        # Display the number of unique categories in all object columns
        categorical_columns = d_tt.select_dtypes(include=['object']).columns
        
        # Create a DataFrame to display the results
        unique_categories = pd.DataFrame({
            'Column': categorical_columns,
            'Unique Categories': [d_tt[col].nunique() for col in categorical_columns]
        })
        
        unique_categories = unique_categories.sort_values(by='Unique Categories', ascending=False)
        unique_categories

## Machine Learning Steps
8. Model Building
- model using Logistic Regression and Random Forest as examples.
  
      # Split the data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
      
      # Logistic Regression model
      logreg = LogisticRegression()
      logreg.fit(X_train, y_train)
      
      # Random Forest model
      rf = RandomForestClassifier(random_state=42)
      rf.fit(X_train, y_train)
  
- model using CatBoostClassifier.
  
          # Splitting the training data into train and validation sets
          X_train, X_val, Y_train, Y_val = train_test_split(train, churn, test_size=0.2, random_state=101)
          # Initializing and training the CatBoostClassifier
          classifier = CatBoostClassifier(loss_function='Logloss', learning_rate=0.01, use_best_model=True, eval_metric='Logloss')
          classifier.fit(X_train, Y_train, eval_set=(X_val, Y_val), early_stopping_rounds=100, verbose=100)

9. 

  

https://github.com/EngrIBGIT/EExpresso-Churn-Prediction-/blob/main/Ibrahim_Notebook_Expresso_Churn.ipynb
https://drive.google.com/file/d/1GmELQx3ZiIaBf0Xm7a1OzAr-it8W_r-W/view?usp=sharing
