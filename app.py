import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import os
import re

# Set up the Streamlit app
st.set_page_config(page_title="Job Recommendation System", page_icon="📈", layout="wide")

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

# Load models 
@st.cache_resource
def load_models():
    with open('description.pkl', 'rb') as f:
        description_model = pickle.load(f)

    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    with open('forest_model.pkl', 'rb') as f:
        forest_model = pickle.load(f)

    return description_model, knn_model, forest_model

description_model, knn_model, forest_model = load_models()

# Load dataset for KNN recommendations
df = pd.read_csv('postings.csv')

# Load TF-IDF matrix and vectorizer
with open('tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load job titles and IDs for dropdowns
title_df = pd.read_csv('title_list.csv')
job_titles = title_df['title'].tolist()

job_id_df = pd.read_csv('job_id_list.csv')
job_ids = job_id_df['job_id'].tolist()

# Add custom CSS for styling from an external GitHub file
st.markdown(
    """
    <style>
    @import url('https://raw.githubusercontent.com/your-username/your-repository/main/styles.css');
    </style>
    """,
    unsafe_allow_html=True
)

# Set up the Streamlit app layout
def display_header(title, image_url):
    st.markdown(f'<h1 class="title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<img src="{image_url}" alt="Header Image" style="width: 80%; max-width: 1200px; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)

# Sidebar for selecting page and user input
st.sidebar.header("User Profile")

# Profile photo upload
uploaded_photo = st.sidebar.file_uploader("Upload a profile photo (JPG, PNG, max 5MB):", type=['jpg', 'png'])
if uploaded_photo is not None:
    # Check file size
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
    display_header('Update Your Profile', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)

    if save_button:
        st.markdown(f'<div class="success-message">Username "{username}" saved successfully!</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Job Recommendations Page
elif page == 'Job Recommendations':
    display_header('Job Recommendations', header_image_url)
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
                file_content = ''
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    file_content += page.extract_text()
                file_df = pd.DataFrame({'description': file_content.split('\n')})

            if 'description' in file_df.columns:
                descriptions = file_df['description'].tolist()
                recommendations = [recommend_jobs(desc) for desc in descriptions]
                st.write('Recommended Jobs:')
                for rec in recommendations:
                    st.write(rec[['title', 'company_name', 'location']])
            else:
                st.markdown('<div class="error-message">The uploaded file must contain a "description" column.</div>', unsafe_allow_html=True)

    elif option == 'Recommend Jobs Based on Job ID':
        st.subheader('Job Recommendations Based on Job ID')
        selected_job_id = st.selectbox('Select Job ID:', options=job_ids)
        
        if st.button('Get Recommendations'):
            if selected_job_id is not None:
                job_id_index = job_ids.index(selected_job_id)
                if 0 <= job_id_index < len(df):
                    distances, indices = knn_model.kneighbors([tfidf_matrix[job_id_index]])
                    similar_jobs = df.iloc[indices.flatten()]
                    st.write('Recommended Jobs:')
                    st.write(similar_jobs[['title', 'company_name', 'location']])
                else:
                    st.markdown('<div class="error-message">Invalid Job ID selected.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">Please select a job ID.</div>', unsafe_allow_html=True)

    elif option == 'Recommend Jobs Based on Title Filter':
        st.subheader('Job Recommendations Based on Title Filter')
        title_filter = st.selectbox('Select Job Title Filter:', options=job_titles)
        
        if st.button('Get Recommendations'):
            if title_filter:
                filtered_jobs = df[df['title'].str.contains(title_filter, case=False, na=False)]
                st.write('Filtered Jobs:')
                st.write(filtered_jobs[['title', 'company_name', 'location']])
            else:
                st.markdown('<div class="error-message">Please select a job title filter.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Predict Candidate Interest Page
elif page == 'Predict Candidate Interest':
    display_header('Predict Candidate Interest', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)
    st.subheader('Predict Candidate Interest', anchor='subtitle')

    # Inputs for predictor model
    views = st.number_input('Views', min_value=0)
    description_length = st.number_input('Description Length', min_value=0)
    average_salary = st.number_input('Average Salary', min_value=0.0, format="%.2f")
    formatted_experience_level = st.selectbox('Experience Level', ['Entry', 'Mid', 'Senior', 'Executive'])
    days_since_listed = st.number_input('Days Since Listed', min_value=0)
    work_type = st.selectbox('Work Type', ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship'])

    if st.button('Predict Interest'):
        if forest_model:
            input_data = pd.DataFrame({
                'views': [views],
                'description_length': [description_length],
                'average_salary': [average_salary],
                'formatted_experience_level': [formatted_experience_level],
                'days_since_listed': [days_since_listed],
                'work_type': [work_type]
            })
            prediction = forest_model.predict(input_data)
            st.write(f'Predicted Interest: {"Interested" if prediction[0] == 1 else "Not Interested"}')
        else:
            st.markdown('<div class="error-message">The predictor model could not be loaded.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Feedback Page
elif page == 'Feedback':
    display_header('Feedback', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)
    st.subheader('Feedback', anchor='subtitle')

    name = st.text_input('Name')
    email = st.text_input('Email')
    feedback = st.text_area('Feedback')

    if st.button('Submit'):
        if name and email and feedback:
            st.markdown('<div class="success-message">Thank you for your feedback!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">Please fill in all fields before submitting.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Function to recommend jobs based on a job description
def recommend_jobs(job_description):
    job_description_tfidf = vectorizer.transform([job_description])
    similarities = cosine_similarity(job_description_tfidf, tfidf_matrix)
    similar_indices = similarities.argsort().flatten()[-10:]
    similar_jobs = df.iloc[similar_indices][::-1]  # Reverse to show most similar first
    return similar_jobs
