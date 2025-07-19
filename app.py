import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model and Data ---
@st.cache_data
def load_data_and_model():
    """
    Loads the dataset and the trained machine learning model.
    """
    df = pd.read_csv("heart-disease.csv")
    model = joblib.load("heart_disease_model.joblib")
    return df, model

df, model = load_data_and_model()

# --- App TItle and Description ---
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("""
This interactive web application utilizes a machine learning model to predict the likelihood of a patient having heart disease. 
You can input patient data in the sidebar to get a prediction, or explore the dataset and model insights using the tabs below.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Patient Data Input")

def user_input_features():
    """
    Creates sidebar widgets for user to input patient data.
    """
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex', (0, 1), format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', (0, 1, 2, 3))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 131)
    chol = st.sidebar.slider('Serum Cholestoral (chol)', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (restecg)', (0, 1, 2))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', (0, 1))
    oldpeak = st.sidebar.slider('ST depression induced by exercise (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the peak exercise ST segment (slope)', (0, 1, 2))
    ca = st.sidebar.selectbox('Number of major vessels colored by flourosopy (ca)', (0, 1, 2, 3, 4))
    thal = st.sidebar.selectbox('Thalium Stress Test Result (thal)', (0, 1, 2, 3))
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Prediction Display ---
st.sidebar.markdown("---")
if st.sidebar.button('Get Prediction', type="primary"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The model predicts this patient **HAS** Heart Disease.", icon="💔")
    else:
        st.success("The model predicts this patient **DOES NOT HAVE** Heart Disease.", icon="❤️")

    st.subheader('Prediction Probability')
    proba_df = pd.DataFrame(prediction_proba, columns=['No Disease', 'Has Disease'], index=['Probability'])
    st.write(proba_df)

# --- Main Page with Tabs for Visualizations ---
tab1, tab2, tab3 = st.tabs(["📊 Data Insights", "🧠 Model Interpretation", "ℹ️ About"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown("Explore the relationships between different patient attributes and heart disease.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Heart Disease Frequency")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.countplot(x='target', data=df, ax=ax1, palette=['#34A853', '#EA4335'])
        ax1.set_title("Distribution of Heart Disease")
        ax1.set_xlabel("0 = No Disease, 1 = Disease")
        ax1.set_ylabel("Patient Count")
        st.pyplot(fig1)

    with col2:
        st.subheader("Disease Frequency by Gender")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.countplot(x='sex', hue='target', data=df, ax=ax2, palette="viridis")
        ax2.set_title("Heart Disease Frequency by Gender")
        ax2.set_xlabel("0 = Female, 1 = Male")
        ax2.set_ylabel("Patient Count")
        plt.legend(title='Heart Disease', labels=['No', 'Yes'])
        st.pyplot(fig2)

    st.subheader("Correlation Matrix")
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

with tab2:
    st.header("Understanding the Model's Predictions")
    st.markdown("The feature importance plot below shows which patient attributes most significantly influence the model's predictions.")
    
    # Feature Importance
    # The model from the notebook is a Logistic Regression, so we use its coefficients.
    if hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'coef_'):
        # For GridSearchCV or RandomizedSearchCV object
        coefficients = model.best_estimator_.coef_[0]
    elif hasattr(model, 'coef_'):
        # For a simple LogisticRegression model
        coefficients = model.coef_[0]
    else:
        st.warning("Could not extract feature importances for this model type.")
        coefficients = np.zeros(len(df.columns[:-1]))

    feature_importance = pd.DataFrame({'feature': df.columns[:-1], 'importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('importance', ascending=True)

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
    ax4.set_xlabel("Importance (Absolute Coefficient Value)")
    ax4.set_ylabel("Feature")
    ax4.set_title("Feature Importance from Logistic Regression Model")
    st.pyplot(fig4)

with tab3:
    st.header("About This Project")
    st.markdown("""
    This application is a practical assignment demonstrating how to deploy a machine learning model as an interactive web service.

    - **Data Source:** The dataset used is the 'Heart Disease UCI' dataset, sourced from Kaggle and the UCI Machine Learning Repository.
    - **Model:** The prediction model is a **Logistic Regression** classifier, which was tuned using GridSearchCV to find the optimal hyperparameters.
    - **Purpose:** To provide a user-friendly interface for predicting heart disease and to offer insights into the data and model behavior, making complex machine learning outputs accessible to a wider audience.
    """)
    st.markdown("---")
    st.write("Developed as a learning exercise based on the provided assignment resources.")

