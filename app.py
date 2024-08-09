import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load necessary resources
def load_resources():
    global recommender_df, title_list_df, knn, X_features, job_ids, vectorizer, tfidf_matrix
    try:
        # Load recommender data
        recommender_df = pd.read_csv('recommender_df.csv')
        title_list_df = pd.read_csv('title_list.csv')
        
        # Prepare KNN model
        X_features = recommender_df[['views', 'applies', 'average_salary']].values
        job_id_list_df = pd.read_csv('job_id_list.csv')
        job_ids = job_id_list_df['job_id'].tolist()
        knn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_features)
        
        # Load TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        tfidf_matrix = vectorizer.transform(recommender_df['processed_description'])
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

def get_top_jobs(title_input, top_n):
    try:
        filtered_df = recommender_df[recommender_df['processed_title'].str.contains(title_input, case=False, na=False)]
        if not filtered_df.empty:
            recommendations = filtered_df.sort_values(by='views', ascending=False).head(top_n)
            st.subheader("Top Recommended Jobs:")
            st.write(recommendations[['processed_title', 'processed_company_name', 'processed_location', 'views']])
            
            with open('recommended_jobs.pkl', 'wb') as f:
                pickle.dump(recommendations, f)
            st.info("Recommendations saved to 'recommended_jobs.pkl'.")
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

    # Header image URL
    header_image_url = "https://github.com/user-attachments/assets/e4b4502f-f99e-4dce-ad20-122843029701"
    st.image(header_image_url, use_column_width=True)

    load_resources()

    # Sidebar for profile and navigation
    st.sidebar.title("Profile and Navigation")

    # Profile picture upload
    profile_picture = st.sidebar.file_uploader("Upload your profile photo", type=["jpg", "jpeg", "png"])
    if profile_picture:
        st.sidebar.image(profile_picture, use_column_width=True, caption="Profile Picture")

    # CV upload
    cv_file = st.sidebar.file_uploader("Upload your CV", type=["pdf", "doc", "docx"])
    if cv_file:
        st.sidebar.download_button(
            label="Download CV",
            data=cv_file,
            file_name="CV_" + cv_file.name,
            mime="application/octet-stream"
        )

    # Navigation options
    options = ["Title-Based Recommendations", "Similar Jobs", "Popular Jobs", "Feedback"]
    selection = st.sidebar.radio("Go to", options)

    if selection == "Title-Based Recommendations":
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
        st.header("Similar Jobs")
        selected_job_id = st.selectbox("Select Job ID to get recommendations:", job_ids)
        if st.button("Get Recommendations"):
            display_knn_recommendations(selected_job_id)

    elif selection == "Popular Jobs":
        st.header("Popular Jobs")
        input_desc = st.text_area("Enter job description to find recommendations:")
        if st.button("Get Recommendations"):
            if input_desc.strip():
                recommended_jobs = recommend_jobs(input_desc)
                if not recommended_jobs.empty:
                    st.subheader("Recommended Jobs:")
                    st.write(recommended_jobs[['processed_title', 'processed_company_name', 'processed_location']])
                else:
                    st.write("No recommendations found.")
            else:
                st.warning("Job description cannot be empty. Please enter a valid description.")

    elif selection == "Feedback":
        st.header("Feedback")
        st.write("We value your feedback! Please let us know your thoughts and suggestions.")
        feedback = st.text_area("Enter your feedback here:")
        if st.button("Submit Feedback"):
            if feedback.strip():
                # Save feedback to a file
                with open('feedback.txt', 'a') as f:
                    f.write(feedback + '\n')
                st.success("Thank you for your feedback!")
            else:
                st.warning("Feedback cannot be empty. Please enter your comments.")

if __name__ == "__main__":
    main()
