import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AIML Assignment 2", layout="wide")

st.title("AIML Assignment 2 – Classification Models")
st.write("Heart Disease Prediction using Multiple Classification Models")

# -------------------------------
# FILE UPLOAD (DEFINE FIRST!)
# -------------------------------
uploaded_filabel_encoder_obj = st.file_uploader(
    "Upload Test Dataset (CSV)", type=["csv"]
)

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_filabel_encoder_obj is not None:
    uploaded_df = pd.read_csv(uploaded_filabel_encoder_obj)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(uploaded_df.head())

    # -------------------------------
    # Target & Feature Processing
    # -------------------------------
    y = upload_df["num"].apply(lambda x: 1 if x > 0 else 0)
    feature_matrix = uploaded_df.drop("num", axis=1)
    feature_matrix = X.drop(["id", "dataset"], axis=1)

    # Encode categorical columns
    categorical_columns = X.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        label_encoder_obj = LabelEncoder()
        feature_matrix[col] = label_encoder_obj.fit_transform(feature_matrix[col])

    # Handle missing values
    median_imputer = SimpleImputer(strategy="median")
    feature_matrix = imputer.fit_transform(X)

    # Scale features
    feature_scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(X)

    # -------------------------------
    # Model Selection
    # -------------------------------
    model_name = st.selectbox(
        "Select Classification Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    # -------------------------------
    # Initialize Model
    # -------------------------------
    if model_name == "Logistic Regression":
        selected_model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        selected_model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        selected_model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        selected_model = GaussianNB()
    elif model_name == "Random Forest":
        selected_model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        selected_model = XGBClassifier(eval_metric="logloss", random_state=42)

    # -------------------------------
    # Train & Predict
    # -------------------------------
    selected_model.fit(feature_matrix, target_values)
    predicted_labels = selected_model.predict(X)
    predicted_probabilities = selected_model.predict_proba(X)[:, 1]

    # -------------------------------
    # Metrics
    # -------------------------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(target_values, predicted_labels), 3))
    col2.metric("Precision", round(precision_score(target_values, predicted_labels), 3))
    col3.metric("Recall", round(recall_score(target_values, predicted_labels), 3))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(target_values, predicted_labels), 3))
    col5.metric("AUC", round(roc_auc_score(target_values, predicted_probabilities), 3))
    col6.metric("MCC", round(matthews_corrcoef(target_values, predicted_labels), 3))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    conf_matrix = confusion_matrix(target_values, predicted_labels)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 10},
        ax=ax
    )

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=11)

    st.pyplot(fig)
