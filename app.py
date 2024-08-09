import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import re

# Helper functions to load and preprocess data
def load_data():
    # Dummy function to simulate data loading
    return pd.DataFrame()

def load_job_ids():
    # Dummy function to simulate loading job IDs
    return ['Job_ID_1', 'Job_ID_2', 'Job_ID_3']

def load_recommender_resources():
    global vectorizer, tfidf_matrix, recommender_df
    # Dummy function to simulate loading resources
    vectorizer = None
    tfidf_matrix = None
    recommender_df = pd.DataFrame({
        'processed_title': ['Data Scientist', 'Software Engineer', 'Data Analyst'],
        'processed_company_name': ['Company A', 'Company B', 'Company C'],
        'processed_location': ['New York', 'San Francisco', 'Boston'],
        'views': [120, 150, 100]
    })

def preprocess_data(df):
    # Dummy function to simulate data preprocessing
    return df

def train_model(X_train, y_train):
    # Dummy function to simulate model training
    return RandomForestRegressor().fit(X_train, y_train)

def preprocess_text(text):
    # Dummy function to simulate text preprocessing
    return text

def get_top_jobs(title_input, top_n):
    try:
        filtered_df = recommender_df[recommender_df['processed_title'].str.contains(title_input, case=False, na=False)]
        if not filtered_df.empty:
            recommendations = filtered_df.sort_values(by='views', ascending=False).head(top_n)
            st.subheader("Top Recommended Jobs:")
            st.write(recommendations[['processed_title', 'processed_company_name', 'processed_location', 'views']])
        else:
            st.warning(f"No jobs found with the title containing '{title_input}'")
    except Exception as e:
        st.error(f"Error getting top jobs: {e}")

def display_knn_recommendations(job_id):
    # Dummy function to simulate KNN recommendations
    st.write(f"KNN recommendations for {job_id}")

def recommend_jobs(input_description, top_n=10):
    input_description_processed = preprocess_text(input_description)
    input_vector = vectorizer.transform([input_description_processed])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    indices = similarities.argsort()[-top_n:][::-1]
    return recommender_df.iloc[indices]

# Streamlit App
def main():
    st.sidebar.title("User Profile")
    st.sidebar.write("Username: Jane Doe")
    st.sidebar.write("Member Since: 2021")
    st.sidebar.image("user_profile.png", width=100)  # Replace with the actual path to the profile image

    # Sidebar navigation
    page = st.sidebar.selectbox("Navigate", ["Home", "Job Application Prediction", "Top Jobs Recommendation", "KNN Recommendations"])
    
    # Next page button
    next_pages = {
        "Home": "Job Application Prediction",
        "Job Application Prediction": "Top Jobs Recommendation",
        "Top Jobs Recommendation": "KNN Recommendations",
        "KNN Recommendations": "Home"
    }
    
    if st.sidebar.button("Next Page"):
        page = next_pages[page]

    # Display the selected page
    if page == "Home":
        st.header("Welcome to MatchWise!")
        st.image("image.png", caption="Welcome Image")  # Replace with the actual path to the image
        st.write("This is the home page where you can get started.")

    elif page == "Job Application Prediction":
        st.header("Predict Job Application Likelihood")
        st.image("image.png", caption="Prediction Image")  # Replace with the actual path to the image
        # Include the Job Application Prediction code here

    elif page == "Top Jobs Recommendation":
        st.header("Top Jobs Recommendation")
        st.image("image.png", caption="Top Jobs Image")  # Replace with the actual path to the image
        load_recommender_resources()
        title_input = st.text_input("Enter Job Title")
        top_n = st.slider("Select Number of Top Jobs", min_value=1, max_value=10, value=5)
        
        if st.button("Recommend Jobs"):
            get_top_jobs(title_input, top_n)

    elif page == "KNN Recommendations":
        st.header("KNN Job Recommendations")
        st.image("image.png", caption="KNN Recommendations Image")  # Replace with the actual path to the image
        load_recommender_resources()
        job_id_list = load_job_ids()
        selected_job_id = st.selectbox("Select Job ID for KNN Recommendations", options=job_id_list)

        if st.button("Get KNN Recommendations"):
            display_knn_recommendations(selected_job_id)

if __name__ == "__main__":
    main()
