import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

# CSS for background color and animations
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        transition: all 0.5s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit application title
st.title('📊 General Data Analysis 🔍')
st.markdown("### Upload your data and view the analysis! 🚀")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df

    df = load_data(uploaded_file)

    # Show the first few rows of the dataset
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Show data types and missing values
    st.subheader("🔍 Data Types and Missing Values")
    st.write(df.dtypes)
    missing_data = df.isnull().sum()
    st.write(missing_data[missing_data > 0])

    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns

    # Fill missing values in numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    # Fill missing values in categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    # Convert categorical data to numeric
    le = LabelEncoder()
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column].astype(str))

    # Select target variable
    target_column = st.selectbox("Select the target variable:", df.columns)

    # Data preprocessing
    X = df.drop([target_column], axis=1)
    y = df[target_column]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # PCA visualization
    st.subheader('PCA Visualization')
    fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=y)
    fig.update_layout(title='PCA: 3D Visualization', scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
    st.plotly_chart(fig)

    # Correlation matrix
    st.subheader('🔗 Correlation Matrix')
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    fig.update_layout(title='Correlation Matrix')
    st.plotly_chart(fig)

    # Model selection and training
    is_classification = len(np.unique(y)) <= 10  # Consider it a classification problem if the number of classes is less than or equal to 10
    
    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Model performance
    st.subheader('🎯 Model Performance')
    if is_classification:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        
        # Confusion Matrix
        st.subheader('🧩 Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, aspect="auto")
        fig.update_layout(title='Confusion Matrix', xaxis_title="Predicted Class", yaxis_title="Actual Class")
        st.plotly_chart(fig)
        
        # Classification Report
        st.subheader('📊 Classification Report')
        st.text(classification_report(y_test, y_pred))
    else:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²) Score: {r2:.2f}")

    # Feature importance visualization
    st.subheader('🏆 Feature Importance Plot')
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    fig = px.bar(feature_importance.head(15), x='importance', y='feature', orientation='h')
    fig.update_layout(title='Top 15 Important Features', xaxis_title="Importance", yaxis_title="Feature")
    st.plotly_chart(fig)

    # Distribution plot of the target variable
    st.subheader('📉 Target Variable Distribution')
    fig = px.histogram(df, x=target_column)
    fig.update_layout(title=f'{target_column} Distribution', xaxis_title=target_column, yaxis_title="Frequency")
    st.plotly_chart(fig)

    # Relationship between two important features
    st.subheader('🔗 Feature Relationship')
    top_features = feature_importance['feature'].head(2).tolist()
    fig = px.scatter(df, x=top_features[0], y=top_features[1], color=target_column)
    fig.update_layout(title=f'Relationship between {top_features[0]} and {top_features[1]}', xaxis_title=top_features[0], yaxis_title=top_features[1])
    st.plotly_chart(fig)

else:
    st.info('👆 Please upload a CSV file.')

# Footer
st.markdown("---")
st.markdown("🎈 This application was created with Streamlit. | Developer: [Your Name]")
