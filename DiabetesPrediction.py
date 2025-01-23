# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
print("Current Working Directory:", os.getcwd())

# Load the dataset
data = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset diabetes\\diabetes.csv')



# Exploratory Data Analysis (EDA)
print("Dataset Overview:")
print(data.info())
print("\nDataset Summary:")
print(data.describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for key features
sns.pairplot(data, vars=['Age', 'BMI', 'Glucose'], hue='Outcome')
plt.show()

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

# Data Preprocessing
# Feature scaling
features = data.drop(columns=['Outcome'])
target = data['Outcome']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Model Development
# 1. Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)


# 2. Support Vector Machine (SVM)
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# Evaluation
print("\nDecision Tree Classifier Report:")
print(classification_report(y_test, dt_preds))

print("\nSupport Vector Machine Report:")
print(classification_report(y_test, svm_preds))
