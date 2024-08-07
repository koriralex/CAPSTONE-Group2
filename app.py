import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import re

# Function to load models with error handling
@st.cache_resource
def load_models():
    try:
        with open('description.pkl', 'rb') as f:
            description_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading description model: {e}")
        description_model = None

    try:
        with open('knn_model.pkl', 'rb') as f:
            knn_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading KNN model: {e}")
        knn_model = None

    try:
        with open('forest_model.pkl', 'rb') as f:
            forest_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading predictor model: {e}")
        forest_model = None

    return description_model, knn_model, forest_model

description_model, knn_model, forest_model = load_models()

# Load dataset for KNN recommendations
df = pd.read_csv('postings.csv')  # Ensure df is defined

# Load TF-IDF matrix and vectorizer
try:
    with open('tfidf_matrix.pkl', 'rb') as file:
        tfidf_matrix = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
except Exception as e:
    st.error(f"Error loading TF-IDF matrix or vectorizer: {e}")
    tfidf_matrix = None
    vectorizer = None

# Load job titles for dropdown
try:
    title_df = pd.read_csv('title_list.csv')
    job_titles = title_df['title'].tolist()
except Exception as e:
    st.error(f"Error loading job titles: {e}")
    job_titles = []

# Load job IDs for dropdown
try:
    job_id_df = pd.read_csv('job_id_list.csv')
    job_ids = job_id_df['job_id'].tolist()
except Exception as e:
    st.error(f"Error loading job IDs: {e}")
    job_ids = []

# Set up the Streamlit app
st.title('Job Recommendation System')

# Sidebar for selecting page
page = st.sidebar.selectbox('Select Page', 
                            ['Job Recommendations',
                             'Predict Candidate Interest'])

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

# Function to recommend jobs based on description
def recommend_jobs(input_description, top_n=10):
    input_description_processed = preprocess_text(input_description)
    input_vector = vectorizer.transform([input_description_processed])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[indices]

# Job Recommendations Page
if page == 'Job Recommendations':
    st.sidebar.title('Recommendation Options')
    option = st.sidebar.selectbox('Select Recommendation Type', 
                                  ['Recommend Jobs Based on Description',
                                   'Recommend Jobs Based on Job ID',
                                   'Recommend Jobs Based on Title Filter'])

    # Job Recommendations Based on Description
    if option == 'Recommend Jobs Based on Description':
        st.subheader('Job Recommendations Based on Description')

        # File upload
        uploaded_file = st.file_uploader("Upload a file with job descriptions (CSV, TXT, or PDF):", type=['csv', 'txt', 'pdf'])

        if uploaded_file is not None:
            # Determine the file type and read the content
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

            # Process descriptions and make recommendations
            if 'description' in file_df.columns:
                descriptions = file_df['description'].tolist()
                recommendations = [recommend_jobs(desc) for desc in descriptions]
                st.write('Recommended Jobs:')
                for rec in recommendations:
                    st.write(rec[['title', 'company_name', 'location']])
            else:
                st.error('The uploaded file must contain a "description" column.')

        else:
            st.info('Please upload a file to get recommendations.')

    # Job Recommendations Based on Job ID (KNN Model)
    elif option == 'Recommend Jobs Based on Job ID':
        st.subheader('Job Recommendations Based on Job ID')
        selected_job_id = st.selectbox('Select Job ID:', options=job_ids)
        
        if st.button('Get Recommendations'):
            if selected_job_id is not None:
                job_id_index = job_ids.index(selected_job_id)
                if 0 <= job_id_index < len(df):
                    distances, indices = knn_model.kneighbors([df.iloc[job_id_index][['views', 'applies', 'average_salary']].values])
                    recommendations = df.iloc[indices[0]]
                    st.write('Recommended Jobs:')
                    st.write(recommendations[['title', 'company_name', 'location']])
                else:
                    st.error('Invalid Job ID. Please select a valid ID.')
            else:
                st.error('Please select a job ID.')

    # Job Recommendations Based on Title Filter
    elif option == 'Recommend Jobs Based on Title Filter':
        st.subheader('Job Recommendations Based on Title Filter')
        selected_title = st.selectbox('Select Job Title:', options=job_titles)
        num_recommendations = st.number_input('Enter the number of top jobs to recommend:', min_value=1, max_value=20, value=5)

        if st.button('Get Recommendations'):
            if selected_title:
                filtered_jobs = df[df['title'] == selected_title]
                st.write('Top recommended jobs based on your input:')
                st.write(filtered_jobs.head(num_recommendations))
            else:
                st.error('Please select a job title.')

# Predict Candidate Interest Page
elif page == 'Predict Candidate Interest':
    st.subheader('Predict Candidate Interest')

    title = st.text_input('Job Title:', key='job_title')
    description = st.text_area('Job Description:', key='job_description')
    location = st.text_input('Location:', key='job_location')
    company_name = st.text_input('Company Name:', key='company_name')
    views = st.number_input('Views:', min_value=0, step=1, key='job_views')
    description_length = st.number_input('Description Length:', min_value=0, step=1, key='description_length')
    average_salary = st.number_input('Average Salary:', min_value=0, step=1, key='job_salary')
    formatted_experience_level = st.selectbox('Experience Level:', ['Entry-level', 'Mid-level', 'Senior-level', 'Manager'], key='experience_level')
    days_since_listed = st.number_input('Days Since Listed:', min_value=0, step=1, key='days_since_listed')
    work_type = st.selectbox('Work Type:', ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship'], key='job_work_type')

    if st.button('Predict Interest'):
        if title and description and location and company_name:
            # Preprocess the text fields
            title_processed = preprocess_text(title)
            description_processed = preprocess_text(description)
            location_processed = preprocess_text(location)
            company_name_processed = preprocess_text(company_name)

            # Create input feature array for prediction
            input_features = pd.DataFrame({
                'title': [title_processed],
                'description': [description_processed],
                'location': [location_processed],
         
