import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import PyPDF2
import io
import os
import re
import numpy as np

# Set up the Streamlit app
st.set_page_config(page_title="MatchWise", page_icon="📈", layout="wide")

# Define the path to save profile photos
UPLOAD_DIR = 'profile_photos'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Function to save uploaded photo
def save_uploaded_file(uploaded_file, user_id):
    file_path = os.path.join(UPLOAD_DIR, f'{user_id}_{uploaded_file.name}')
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())
    return file_path

# Load models with error handling
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'recommended_jobs': 'recommended_jobs.pkl',
        'knn_model': 'knn_model.pkl',
        'random_forest_model': 'random_forest_model.pkl'
    }
    for key, filename in model_files.items():
        try:
            with open(filename, 'rb') as f:
                models[key] = pickle.load(f)
        except Exception as e:
            models[key] = None
            st.error(f"Error loading {filename}: {e}")
    return models

models = load_models()
recommended_jobs, knn_model, forest_model = models['recommended_jobs'], models['knn_model'], models['random_forest_model']

# Load dataset and TF-IDF matrix/vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    recommender_df = pd.read_csv('recommender_df.csv')  # Load recommender_df
    predictor_df = pd.read_csv('predictor_df.csv')  # Load predictor_df
except Exception as e:
    vectorizer, recommender_df, predictor_df = None, None, None
    st.error(f"Error loading TF-IDF vectorizer, recommender_df, or predictor_df: {e}")

# Load job titles and IDs for dropdowns
job_titles, job_ids = [], []
try:
    title_df = pd.read_csv('title_list.csv')
    job_titles = title_df['title'].tolist()
except Exception as e:
    st.error(f"Error loading job titles: {e}")

try:
    job_id_df = pd.read_csv('job_id_list.csv')
    job_ids = job_id_df['job_id'].tolist()
except Exception as e:
    st.error(f"Error loading job IDs: {e}")

# Path to postings CSV file
file_path = r'C:\Users\Caro\Downloads\postings.csv'

# Add custom CSS for styling from an external GitHub file
st.markdown(
    """
    <style>
    @import url('https://raw.githubusercontent.com/ge_saka/CAPSTONE-Group2/main/styles.css');
    </style>
    """,
    unsafe_allow_html=True
)

# Set up the Streamlit app layout
def display_header(title, image_url):
    st.markdown(f'<h1 class="title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<img src="{image_url}" alt="Header Image" style="width: 80%; max-width: 1200px; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)

# Sidebar for selecting page and user input
st.sidebar.header("MatchWise - User Profile")

# Profile photo upload
uploaded_photo = st.sidebar.file_uploader("Upload a profile photo (JPG, PNG, max 5MB):", type=['jpg', 'png'])
if uploaded_photo is not None:
    file_size = len(uploaded_photo.read())  # Read the entire file to get its size
    uploaded_photo.seek(0)  # Reset file pointer to the beginning after reading

    if file_size > 5 * 1024 * 1024:  # 5MB limit
        st.sidebar.error("The file size should not exceed 5MB.")
    else:
        user_id = 'example_user_id'  # Replace with actual user ID if available
        photo_path = save_uploaded_file(uploaded_photo, user_id)
        st.sidebar.image(photo_path, caption='Uploaded Profile Photo', use_column_width=True, output_format='JPEG')

# Username
username = st.sidebar.text_input("Username")

# Email
email = st.sidebar.text_input("Email")

# Email validation
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

email_valid = is_valid_email(email)
if not email_valid and email:
    st.sidebar.markdown('<div class="error-message">Please enter a valid email address.</div>', unsafe_allow_html=True)

save_button = st.sidebar.button("Save")

# Image URL
header_image_url = "https://github.com/user-attachments/assets/e4b4502f-f99e-4dce-ad20-122843029701"

# Page selection in the sidebar
page = st.sidebar.selectbox('Select Page', ['Profile Update', 'Job Recommendations', 'Predict Candidate Interest', 'Feedback'])

# Profile Update Page
if page == 'Profile Update':
    display_header('Update Your Profile - MatchWise', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)

    if save_button:
        st.markdown(f'<div class="success-message">Username "{username}" saved successfully!</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Job Recommendations Page
elif page == 'Job Recommendations':
    display_header('Job Recommendations - MatchWise', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)

    option = st.selectbox('Select Recommendation Type', ['Recommend Jobs Based on Description', 'Recommend Jobs Based on Job ID', 'Recommend Jobs Based on Title Filter'])

    if option == 'Recommend Jobs Based on Description':
        st.subheader('Job Recommendations Based on Description')

        uploaded_file = st.file_uploader("Upload a file with job descriptions (CSV, TXT, or PDF):", type=['csv', 'txt', 'pdf'])

        if uploaded_file is not None:
            if uploaded_file.type == 'text/csv':
                file_df = pd.read_csv(uploaded_file)
            elif uploaded_file.type == 'text/plain':
                file_content = uploaded_file.read().decode('utf-8')
                file_df = pd.DataFrame({'description': file_content.split('\n')})
            elif uploaded_file.type == 'application/pdf':
                reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                file_content = ''.join([page.extract_text() for page in reader.pages])
                file_df = pd.DataFrame({'description': file_content.split('\n')})

            if 'description' in file_df.columns:
                descriptions = file_df['description'].tolist()
                recommendations = [recommended_jobs(desc) for desc in descriptions]
                st.write('Recommended Jobs:')
                for rec in recommendations:
                    st.write(rec[['processed_title', 'processed_company_name', 'processed_location']])
            else:
                st.markdown('<div class="error-message">The uploaded file must contain a "description" column.</div>', unsafe_allow_html=True)

    elif option == 'Recommend Jobs Based on Job ID':
        st.subheader('Job Recommendations Based on Job ID')
        selected_job_id = st.selectbox('Select Job ID:', options=job_ids)
        
        if st.button('Get Recommendations'):
            if selected_job_id is not None:
                job_id_index = job_ids.index(selected_job_id)
                
                if recommender_df is not None:
                    if 0 <= job_id_index < len(recommender_df):
                        job_description = recommender_df.iloc[job_id_index]['processed_description']
                        job_vector = vectorizer.transform([job_description])
                        
                        # Ensure KNN model is trained with the right features
                        if knn_model:
                            # Prepare feature matrix
                            X_features = recommender_df[['views', 'applies', 'average_salary']].values

                            # Apply KNN for job recommendations
                            knn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_features)
                            distances, indices = knn.kneighbors(X_features)

                            # Calculate average distance of nearest neighbors
                            average_distance = np.mean(distances)
                            st.write(f"Average Distance to Nearest Neighbors: {average_distance:.2f}")

                            # Display KNN recommendations
                            if 0 <= job_id_index < len(recommender_df):
                                recommendations = indices[job_id_index]
                                top_recommendations = recommender_df.iloc[recommendations.flatten()]
                                st.write('Recommended Jobs:')
                                st.write(top_recommendations[['processed_title', 'processed_company_name', 'processed_location']])
                            else:
                                st.write(f"Job ID {job_id_index} is out of range.")
                        else:
                            st.markdown('<div class="error-message">KNN model is not available.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-message">Job ID index is out of range.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">Recommender DataFrame is not loaded correctly.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">Please select a Job ID.</div>', unsafe_allow_html=True)

    elif option == 'Recommend Jobs Based on Title Filter':
        st.subheader('Job Recommendations Based on Title Filter')
        title_filter = st.selectbox('Select Job Title Filter:', options=job_titles)

        if st.button('Get Recommendations'):
            filtered_jobs = recommender_df[recommender_df['processed_title'].str.contains(title_filter, case=False, na=False)]
            if not filtered_jobs.empty:
                st.write(filtered_jobs[['processed_title', 'processed_company_name', 'processed_location']])
            else:
                st.write("No jobs found matching the title filter.")

    st.markdown('</div>', unsafe_allow_html=True)

# Predict Candidate Interest Page
elif page == 'Predict Candidate Interest':
    display_header('Predict Candidate Interest - MatchWise', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)

    if forest_model:
        # User inputs for prediction
        views = st.number_input('Number of Views', min_value=0, value=0)
        description_length = st.number_input('Description Length', min_value=0, value=0)
        average_salary = st.number_input('Average Salary', min_value=0, value=0)
        experience_level = st.selectbox('Experience Level', options=['Entry Level', 'Mid Level', 'Senior Level'])
        days_since_listed = st.number_input('Days Since Listed', min_value=0, value=0)
        work_type = st.selectbox('Work Type', options=['Full-time', 'Part-time', 'Contract', 'Internship'])

        # Map experience level to numerical value
        experience_mapping = {'Entry Level': 1, 'Mid Level': 2, 'Senior Level': 3}
        experience_value = experience_mapping.get(experience_level, 0)

        # Map work type to numerical value
        work_type_mapping = {'Full-time': 1, 'Part-time': 2, 'Contract': 3, 'Internship': 4}
        work_type_value = work_type_mapping.get(work_type, 0)

        # Create input features DataFrame
        input_features = pd.DataFrame({
            'views': [views],
            'description_length': [description_length],
            'average_salary': [average_salary],
            'formatted_experience_level': [experience_value],
            'days_since_listed': [days_since_listed],
            'work_type': [work_type_value]
        })

        # Predict interest
        prediction_prob = forest_model.predict_proba(input_features)[0][1]  # Probability of class 1
        threshold = 0.5  # Set your threshold here

        if prediction_prob > threshold:
            prediction = 'High'
        else:
            prediction = 'Low'

        st.write(f"Prediction Probability: {prediction_prob:.2f}")
        st.write(f"Predicted Interest Level: {prediction}")

        # Display filtered job postings based on prediction
        if prediction == 'High':
            filtered_jobs = predictor_df[predictor_df['average_salary'] >= average_salary]
        else:
            filtered_jobs = predictor_df[predictor_df['average_salary'] < average_salary]

        if not filtered_jobs.empty:
            st.write('Filtered Job Postings:')
            st.write(filtered_jobs[['job_id', 'description', 'average_salary']])
        else:
            st.write("No job postings match the criteria.")

    else:
        st.markdown('<div class="error-message">Random Forest model is not available.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Feedback Page
elif page == 'Feedback':
    display_header('Feedback - MatchWise', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)
    st.subheader('Provide Your Feedback')
    feedback = st.text_area('Share your feedback or suggestions here:')
    submit_feedback = st.button('Submit Feedback')

    if submit_feedback:
        if feedback:
            st.markdown('<div class="success-message">Thank you for your feedback!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">Please provide feedback before submitting.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
