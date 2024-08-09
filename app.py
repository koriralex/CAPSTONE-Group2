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

# Load dataset and job IDs
def load_data():
    try:
        df = pd.read_csv('postings.csv')
        imputer = SimpleImputer(strategy='median')
        df[['views', 'max_salary', 'min_salary', 'listed_time', 'applies']] = imputer.fit_transform(
            df[['views', 'max_salary', 'min_salary', 'listed_time', 'applies']]
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def load_job_ids():
    try:
        job_id_df = pd.read_csv('job_id_list.csv')
        return job_id_df['job_id'].tolist()
    except Exception as e:
        st.error(f"Error loading job IDs: {e}")
        st.stop()

def load_recommender_resources():
    global recommender_df, title_list_df, knn, X_features, vectorizer, tfidf_matrix
    try:
        recommender_df = pd.read_csv('recommender_df.csv')
        title_list_df = pd.read_csv('title_list.csv')
        
        X_features = recommender_df[['views', 'applies', 'average_salary']].values
        job_id_list_df = pd.read_csv('job_id_list.csv')
        knn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_features)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = vectorizer.transform(recommender_df['processed_description'])
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

def preprocess_data(df):
    df['average_salary'] = (df['max_salary'] + df['min_salary']) / 2
    df = df.drop(columns=['max_salary', 'min_salary', 'description'])
    
    le_work_type = LabelEncoder()
    le_experience_level = LabelEncoder()
    
    df['work_type'] = le_work_type.fit_transform(df['work_type'].astype(str))
    df['formatted_experience_level'] = le_experience_level.fit_transform(df['formatted_experience_level'].astype(str))
    
    X = df[['views', 'average_salary', 'listed_time', 'work_type', 'formatted_experience_level']]
    y = df['applies']
    
    return X, y, le_work_type, le_experience_level, df[['job_id']]

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
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
    try:
        job_index = recommender_df[recommender_df['job_id'] == job_id].index[0]
        distances, indices = knn.kneighbors([X_features[job_index]])
        average_distance = np.mean(distances)
        
        st.write(f"Average Distance to Nearest Neighbors: {average_distance:.2f}")
        
        recommendations = indices.flatten()
        top_recommendations = recommender_df.iloc[recommendations]
        
        st.subheader("KNN Recommendations based on JOB ID:")
        st.write(top_recommendations[['processed_title', 'processed_company_name', 'processed_location']])
    except Exception as e:
        st.error(f"Error displaying recommendations: {e}")

def recommend_jobs(input_description, top_n=10):
    input_description_processed = preprocess_text(input_description)
    input_vector = vectorizer.transform([input_description_processed])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    indices = similarities.argsort()[-top_n:][::-1]
    return recommender_df.iloc[indices]

# Streamlit App
def main():
    st.title("MatchWise: Intelligent Job Matching and Application Trend Prediction")

    header_image_url = "https://github.com/user-attachments/assets/e4b4502f-f99e-4dce-ad20-122843029701"
    st.image(header_image_url, use_column_width=True)

    # Load data and resources
    df = load_data()
    load_recommender_resources()

    # Sidebar for profile and navigation
    st.sidebar.title("Profile and Navigation")

    profile_picture = st.sidebar.file_uploader("Upload your profile photo", type=["jpg", "jpeg", "png"])
    if profile_picture:
        st.sidebar.image(profile_picture, use_column_width=True, caption="Profile Picture")

    cv_file = st.sidebar.file_uploader("Upload your CV", type=["pdf", "doc", "docx"])
    if cv_file:
        st.sidebar.download_button(
            label="Download CV",
            data=cv_file,
            file_name="CV_" + cv_file.name,
            mime="application/octet-stream"
        )

    options = ["Predict Job Application Likelihood", "Title-Based Recommendations", "Similar Jobs", "Popular Jobs", "Feedback"]
    selection = st.sidebar.radio("Go to", options)

    if selection == "Predict Job Application Likelihood":
        st.header("Job Application Prediction")
        
        X, y, le_work_type, le_experience_level, job_ids = preprocess_data(df)
        job_id_list = load_job_ids()
        
        X_train, X_test, y_train, y_test, job_id_train, job_id_test = train_test_split(X, y, job_ids, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = train_model(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        selected_job_id = st.selectbox("Select Job ID", options=job_id_list)
        views = st.slider("Views", min_value=0, max_value=10000, value=500)
        max_salary = st.slider("Maximum Salary", min_value=30000, max_value=200000, value=50000)
        min_salary = st.slider("Minimum Salary", min_value=30000, max_value=200000, value=30000)
        listed_time = st.slider("Listed Time (days)", min_value=0, max_value=365, value=30)
        
        work_type_options = df['work_type'].unique()
        selected_work_type = st.selectbox("Work Type", options=work_type_options)
        
        experience_level_options = df['formatted_experience_level'].unique()
        selected_experience_level = st.selectbox("Formatted Experience Level", options=experience_level_options)

        average_salary = (max_salary + min_salary) / 2

        user_input = pd.DataFrame({
            'job_id': [selected_job_id],
            'views': [views],
            'average_salary': [average_salary],
            'listed_time': [listed_time],
            'work_type': [selected_work_type],
            'formatted_experience_level': [selected_experience_level]
        })

        user_input['work_type'] = le_work_type.transform(user_input['work_type'].astype(str))
        user_input['formatted_experience_level'] = le_experience_level.transform(user_input['formatted_experience_level'].astype(str))
        user_input_scaled = scaler.transform(user_input.drop(columns=['job_id']))

        if st.button("Get Prediction"):
            predicted_applies = model.predict(user_input_scaled)[0]
            st.subheader("Prediction")
            st.write(f"Predicted Number of Applies: {predicted_applies:.2f}")
            st.write(f"Job ID: {selected_job_id}")

    elif selection == "Title-Based Recommendations":
        st.header("Job Title-Based Recommendations")
        title_options = title_list_df['title'].tolist()
        selected_title = st.selectbox("Select a job title:", title_options)
        top_n = st.number_input("Enter the number of top jobs to recommend:", min_value=1, value=10, step=1)
        if st.button("Get Recommendations"):
            if selected_title.strip():
                get_top_jobs(selected_title, top_n)
            else:
                st.warning("Please select a job title.")

    elif selection == "Similar Jobs":
        st.header("Job Similarity Checker")
        job_id_list = load_job_ids()
        job_id_selected = st.selectbox("Select Job ID", options=job_id_list)
        if st.button("Get Similar Jobs"):
            display_knn_recommendations(job_id_selected)

    elif selection == "Popular Jobs":
        st.header("Popular Jobs")
        st.write("Top recommended jobs based on views:")
        top_n = st.number_input("Enter the number of popular jobs to display:", min_value=1, value=5, step=1)
        top_jobs = recommender_df.sort_values(by='views', ascending=False).head(top_n)
        st.write(top_jobs[['processed_title', 'processed_company_name', 'processed_location', 'views']])

    elif selection == "Feedback":
        st.header("Feedback")
        st.text_area("Feedback", "Enter your feedback here...")
        st.button("Submit")

if __name__ == "__main__":
    main()
