<div align="center">

# 🩺 Heart Disease Prediction Dashboard

### An End-to-End Machine Learning Project to Classify Heart Disease Presence with a Deployed Streamlit Web Application

</div>

---

## 🚀 Live Demo

The interactive web application is deployed on Streamlit Community Cloud. You can access it here:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-prediction-dashboard.streamlit.app/)

---

## 🎯 Project Overview

This project aims to leverage machine learning to predict the likelihood of a patient having heart disease based on 13 clinical features, such as age, sex, chest pain type, and cholesterol levels. The primary goal is to build a reliable classification model that can assist medical professionals in making preliminary diagnoses. The project covers the entire machine learning workflow, from data exploration and preprocessing to model training, evaluation, and finally, deployment as a user-friendly web application.

### ✨ Key Features:
- **In-depth EDA:** Comprehensive exploratory data analysis with visualizations to understand the dataset.
- **Model Comparison:** Trained and evaluated three different classification models: Logistic Regression, K-Nearest Neighbors, and Random Forest.
- **Hyperparameter Tuning:** Optimized the best-performing model (Logistic Regression) using `GridSearchCV` to maximize its predictive power.
- **Interactive UI:** A multi-page Streamlit dashboard that allows for:
    - Real-time predictions based on user-input patient data.
    - Visualization of dataset insights and feature correlations.
    - Interpretation of the model's decisions through feature importance analysis.

---

## 🛠️ Technology Stack

| Technology | Description |
| :--- | :--- |
| **Python** | Core programming language for the project. |
| **Pandas & NumPy** | Used for data loading, manipulation, and numerical computation. |
| **Matplotlib & Seaborn** | Employed for creating insightful data visualizations. |
| **Scikit-learn** | The primary library for model training, tuning, and evaluation. |
| **Jupyter Notebook** | For conducting the initial research, analysis, and model experimentation. |
| **Streamlit** | For building and deploying the interactive web application. |
| **Joblib** | Used for serializing and saving the trained machine learning model. |

---

## 📈 Final Model Performance

The final model, a **tuned Logistic Regression classifier**, achieved the following performance on the hold-out test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 89% |
| **Precision (for Class 1)** | 88% |
| **Recall (for Class 1)** | 91% |
| **F1-Score (for Class 1)**| 89% |

The model demonstrates a high recall, which is crucial in a medical context as it correctly identifies 91% of patients who actually have heart disease.

---

## 🚀 How to Run This Project Locally

Follow these steps to set up and run the application on your own machine.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/Heart-Disease-Prediction-App.git](https://github.com/Divyansh-debug/Heart-Disease-Classification-Project.git)
    cd Heart-Disease-Classification-Project
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application will open in a new tab in your web browser.

---

## 📂 Repository Structure

```
├── 📄 app.py                  # Main Streamlit application script
├── 🧠 heart_disease_model.joblib # Trained Logistic Regression model
├── 📝 requirements.txt        # Required Python libraries
├── 📓 end-to-end-heart-disease-classification-final.ipynb # Jupyter Notebook with full analysis
├── 💾 heart-disease.csv       # The dataset
└── 📖 README.md               # This file
```

---

## 📊 Dataset

This project utilizes the **Heart Disease UCI dataset**, sourced from the UCI Machine Learning Repository and made available on Kaggle. It contains 14 attributes collected from 303 patients.

---

## 📄 License

This project is licensed under the **MIT License**.
