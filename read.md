Diabetes Prediction Model
This project is all about predicting diabetes using machine learning. It uses a dataset with various health metrics and a target variable that indicates whether a person has diabetes or not.

What This Project Does
Here’s what this project includes:

Understanding the Data: We explore the dataset, find patterns, and visualize important features.

Preparing the Data: This step involves scaling the data, splitting it into training and testing sets, and handling missing values.

Building Models: We train two machine learning models:

A Decision Tree Classifier

A Support Vector Machine (SVM)

Evaluating the Models: We measure how well each model performs using accuracy, classification reports, and confusion matrices.

What You Need to Run This Project
Libraries
Make sure you have these Python libraries installed:

pandas

numpy

matplotlib

seaborn

scikit-learn

Install them using this command if needed:

pip install pandas numpy matplotlib seaborn scikit-learn
Dataset
The project uses a dataset called diabetes.csv. It contains the following columns:

Pregnancies: Number of pregnancies

Glucose: Plasma glucose concentration

BloodPressure: Blood pressure in mm Hg

SkinThickness: Skinfold thickness in mm

Insulin: Serum insulin levels

BMI: Body mass index

DiabetesPedigreeFunction: Genetic risk factor

Age: Patient's age

Outcome: Whether the person has diabetes (1) or not (0)

Place the dataset in this folder: C:\Users\Dell\Downloads\dataset diabetes\.

How It Works
1. Understanding the Data
We look at the dataset’s structure using info() and describe().

Correlations between features are visualized using a heatmap.

A pairplot shows how key features like Age, BMI, and Glucose relate to diabetes outcomes.

2. Preparing the Data
We check for missing values and fix them if necessary.

Features are standardized using StandardScaler.

The dataset is split into 80% training and 20% testing.

3. Builded two machine learning Models  
Decision Tree Classifier
A simple decision tree model is trained.

It predicts outcomes for the test data.

Support Vector Machine (SVM)
An SVM model with probability estimates is trained.

It predicts outcomes for the test data.

4. Evaluating the Models
We evaluate both models using:

Classification Report: This includes metrics like precision, recall, and F1-score.

Accuracy Score: How accurate the predictions are.

Confusion Matrix: A breakdown of correct and incorrect predictions.

Results
The Decision Tree and SVM models are both evaluated, and the results are displayed in the console.

How to Run
Download the project and place the dataset in the correct folder.

Run the script using this command:

python diabetes_prediction.py
The script will generate visualizations and print model performance metrics in the console.

Visualizations
Correlation Heatmap: Shows relationships between features.

Pairplot: Displays how key features differ based on diabetes status.

