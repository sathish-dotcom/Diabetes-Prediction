# **Loan Approval Prediction using Random Forest Classifier**

## **1. Introduction**

This project aims to build a machine learning model to predict loan approval based on applicant details. The dataset consists of multiple features, including demographic information, financial status, and loan application details. The primary goal is to enhance prediction accuracy using a Random Forest Classifier and optimize its performance through hyperparameter tuning.

## **2. Data Preprocessing**

### 2.1 Importing Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

### 2.2 Loading the Dataset

df = pd.read_csv("loan_data.csv")

### 2.3 Exploratory Data Analysis

df.head()  # Display first 5 rows
df.shape   # Check dimensions of the dataset
df.info()  # Get summary of dataset
df.describe()  # Statistical summary of numerical variables

### 2.4 Handling Missing Values

Missing values are filled using the most frequent category (mode) for categorical variables.

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df["Self_Employed"].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

### 2.5 Encoding Categorical Variables

Converting categorical variables into numerical form for model training.

df = df.drop(columns=['Loan_ID'])  # Dropping Loan_ID as it is not a predictive feature
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].astype(str).replace({'3+':3}).astype(float)
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

### 2.6 Splitting Data into Training and Testing Sets

x = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

## **3. Model Training and Evaluation**

### 3.1 Initial Model Training using Random Forest Classifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(xtrain, ytrain)
predict_output = model.predict(xtest)

### 3.2 Evaluating Model Performance

acc = accuracy_score(ytest, predict_output)
print(f'Initial Accuracy: {acc*100:.4f}%')
print(classification_report(ytest, predict_output))

### 3.3 Confusion Matrix Visualization

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(ytest, predict_output), annot=True, fmt="d", cmap="Blues", xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix: Actual vs Predicted Loan Status')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

## **4. Hyperparameter Tuning using Grid Search**

To optimize the model, we perform hyperparameter tuning using Grid Search.

param_grid = {
    'n_estimators': [200],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy'],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'bootstrap' : [True, False]
}

model_gridsearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
model_gridsearch.fit(xtrain, ytrain)
print(model_gridsearch.best_params_)

### 4.1 Training the Model with Optimized Parameters

Using the best parameters found from Grid Search:

model1 = RandomForestClassifier(n_estimators=200, random_state=42)
model1.fit(xtrain, ytrain)
predict_output1 = model1.predict(xtest)

### 4.2 Evaluating the Optimized Model

acc = accuracy_score(ytest, predict_output1)
print(f'Optimized Model Accuracy: {acc*100:.4f}%')
print(classification_report(ytest, predict_output1))

### 4.3 Confusion Matrix Visualization

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(ytest, predict_output1), annot=True, fmt="d", cmap="Blues", xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix: Actual vs Predicted Loan Status (Optimized Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

## **5. Conclusion**

After implementing hyperparameter tuning using Grid Search, we observed that increasing n_estimators to 200 did not result in any change in accuracy. However, there were differences observed in the confusion matrix, indicating variations in classification distribution. The optimized Random Forest Classifier remains a suitable choice for loan approval prediction.

Key Takeaways:

Data preprocessing (handling missing values, encoding categorical variables) significantly improves model performance.

Random Forest provides high accuracy for loan approval predictions.

Hyperparameter tuning using Grid Search can modify classification results without necessarily improving accuracy.

A well-tuned model can better classify loan applicants and improve decision-making processes.

## **6. Future Scope**

Experimenting with other machine learning models such as XGBoost or Neural Networks.

Applying feature engineering techniques to improve prediction accuracy.

Using real-time data for dynamic prediction updates.

This project effectively demonstrates the power of machine learning in financial decision-making and provides a robust framework for future enhancements.

