# Heart Disease Classification Project

## Project Overview

This project applies multiple machine learning classification techniques
to predict the presence of heart disease. It demonstrates the full
machine learning lifecycle including preprocessing, model training,
evaluation, and deployment using Streamlit.

------------------------------------------------------------------------

## Dataset Details

The UCI Heart Disease dataset is used in this project. It includes
clinical and medical attributes such as:

-   Age
-   Gender
-   Chest pain type
-   Blood pressure
-   Cholesterol level
-   ECG results
-   Maximum heart rate achieved
-   Exercise-induced angina

Target Variable: - num = 0 → No heart disease - num \> 0 → Heart disease
detected (converted into binary classification)

------------------------------------------------------------------------

## Implemented Models

The following classification models are implemented:

1.  Logistic Regression\
2.  Decision Tree Classifier\
3.  K-Nearest Neighbors (KNN)\
4.  Gaussian Naive Bayes\
5.  Random Forest Classifier\
6.  XGBoost Classifier

------------------------------------------------------------------------

## Performance Metrics

Each model is evaluated using:

-   Accuracy\
-   Area Under the Curve (AUC)\
-   Precision\
-   Recall\
-   F1 Score\
-   Matthews Correlation Coefficient (MCC)

------------------------------------------------------------------------

## Streamlit Application

The interactive web application provides:

-   CSV dataset upload functionality\
-   Dataset preview\
-   Model selection option\
-   Real-time evaluation metrics\
-   Confusion matrix visualization

------------------------------------------------------------------------

## Running the Application

1.  Install dependencies: pip install -r requirements.txt

2.  Start the Streamlit server: streamlit run app.py

------------------------------------------------------------------------

## Folder Structure

AIML_Assignment_2/ ├── app.py ├── AIML_Assignment_Notebook.ipynb ├──
requirements.txt ├── README.md └── data/ └── Heart_patietns.csv

------------------------------------------------------------------------

## Summary

This project provides practical exposure to building, evaluating, and
deploying machine learning classification models using Python,
Scikit-learn, XGBoost, and Streamlit.
