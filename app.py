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
                st.markdown('<div class="error-message">Please select a valid Job ID.</div>', unsafe_allow_html=True)

    elif option == 'Recommend Jobs Based on Title Filter':
        st.subheader('Job Recommendations Based on Title Filter')
        selected_title = st.selectbox('Select Job Title:', options=job_titles)
        
        if st.button('Get Recommendations'):
            if selected_title:
                if recommender_df is not None:
                    filtered_jobs = recommender_df[recommender_df['processed_title'] == selected_title]

                    # Exclude problematic columns
                    required_columns = ['job_id', 'processed_title', 'processed_company_name', 'processed_location', 'average_salary']
                    filtered_jobs = filtered_jobs[required_columns]
                    
                    st.write('Recommended Jobs:')
                    st.write(filtered_jobs)
                else:
                    st.markdown('<div class="error-message">Recommender DataFrame is not loaded correctly.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">Please select a valid Job Title.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Predict Candidate Interest Page
elif page == 'Predict Candidate Interest':
    display_header('Predict Candidate Interest - MatchWise', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)

    min_salary = st.number_input('Minimum Salary', value=0)
    max_salary = st.number_input('Maximum Salary', value=1000000)
    work_type = st.selectbox('Work Type', options=['Full-time', 'Part-time', 'Contract', 'Internship'])

    if st.button('Predict'):
        if predictor_df is not None:
            # Filter the predictor_df based on input values
            filtered_jobs = predictor_df[(predictor_df['average_salary'] >= min_salary) &
                                         (predictor_df['average_salary'] <= max_salary) &
                                         (predictor_df['work_type'] == work_type)]
            
            # Exclude problematic columns
            required_columns = ['job_id', 'average_salary']
            filtered_jobs = filtered_jobs[required_columns]
            
            # Predict candidate interest
            if forest_model:
                X = filtered_jobs[['average_salary']].values
                predictions = forest_model.predict(X)
                filtered_jobs['predicted_interest'] = predictions

                st.write('Predicted Candidate Interest:')
                st.write(filtered_jobs[['job_id', 'average_salary', 'predicted_interest']])
            else:
                st.markdown('<div class="error-message">Random Forest model is not available.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">Predictor DataFrame is not loaded correctly.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Feedback Page
elif page == 'Feedback':
    display_header('Feedback - MatchWise', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)

    feedback = st.text_area('Your Feedback')

    if st.button('Submit Feedback'):
        if feedback:
            st.markdown('<div class="success-message">Thank you for your feedback!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">Please provide feedback before submitting.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Additional Features
st.markdown('<div class="footer">Developed by [Your Name]</div>', unsafe_allow_html=True)
