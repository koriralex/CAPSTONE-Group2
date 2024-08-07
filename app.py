import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io

# Load models
@st.cache_resource
def load_models():
    with open('description.pkl', 'rb') as f:
        description_model = pickle.load(f)
    with open('knn.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    with open('predictor.pkl', 'rb') as f:
        predictor_model = pickle.load(f)
    return description_model, knn_model, predictor_model

description_model, knn_model, predictor_model = load_models()

# Load dataset for KNN recommendations
recommender_df = pd.read_csv('postings.csv')  # Adjust path as needed

# Set up the Streamlit app
st.title('Job Recommendation System')

# Sidebar for selecting recommendation method
st.sidebar.title('Recommendation Options')
option = st.sidebar.selectbox('Select Recommendation Type', 
                              ['Recommend Jobs Based on Description',
                               'Recommend Jobs Based on Job ID',
                               'Recommend Jobs Based on Title Filter',
                               'Predict Candidate Interest'])

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
            reader = PyPDF2.PdfFileReader(io.BytesIO(uploaded_file.read()))
            file_content = ''
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                file_content += page.extract_text()
            file_df = pd.DataFrame({'description': file_content.split('\n')})
        
        # Process descriptions and make recommendations
        if 'description' in file_df.columns:
            descriptions = file_df['description'].tolist()
            recommendations = [description_model.predict([desc])[0] for desc in descriptions]
            st.write('Recommended Jobs:')
            st.write(pd.DataFrame(recommendations, columns=['Recommended Jobs']))
        else:
            st.error('The uploaded file must contain a "description" column.')
    
    else:
        st.info('Please upload a file to get recommendations.')

# Job Recommendations Based on Job ID (KNN Model)
elif option == 'Recommend Jobs Based on Job ID':
    st.subheader('Job Recommendations Based on Job ID')
    job_id = st.number_input('Enter Job ID:', min_value=0, step=1)
    
    if st.button('Get Recommendations'):
        if job_id is not None:
            if 0 <= job_id < len(recommender_df):
                distances, indices = knn_model.kneighbors([recommender_df.iloc[job_id][['views', 'applies', 'average_salary']].values])
                recommendations = recommender_df.iloc[indices[0]]
                st.write('Recommended Jobs:')
                st.write(recommendations[['title', 'company_name', 'location']])
            else:
                st.error('Invalid Job ID. Please enter a valid ID.')
        else:
            st.error('Please enter a job ID.')

# Job Recommendations Based on Title Filter
elif option == 'Recommend Jobs Based on Title Filter':
    st.subheader('Job Recommendations Based on Title Filter')
    job_title = st.text_input('Enter job title to filter by:')
    num_recommendations = st.number_input('Enter the number of top jobs to recommend:', min_value=1, max_value=20, value=5)
    
    if st.button('Get Recommendations'):
        if job_title:
            filtered_jobs = recommender_df[recommender_df['title'].str.contains(job_title, case=False)]
            st.write('Top recommended jobs based on your input:')
            st.write(filtered_jobs.head(num_recommendations))
        else:
            st.error('Please enter a job title.')

# Predict Candidate Interest
elif option == 'Predict Candidate Interest':
    st.subheader('Predict Candidate Interest')
    
    title = st.text_input('Job Title:', key='job_title')
    description = st.text_area('Job Description:', key='job_description')
    location = st.text_input('Location:', key='job_location')
    company_name = st.text_input('Company Name:', key='company_name')
    views = st.number_input('Views:', min_value=0, step=1, key='job_views')
    applies = st.number_input('Applies:', min_value=0, step=1, key='job_applies')
    average_salary = st.number_input('Average Salary:', min_value=0, step=1, key='job_salary')
    work_type = st.selectbox('Work Type:', ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship'], key='job_work_type')
    experience_level = st.selectbox('Experience Level:', ['Entry-level', 'Mid-level', 'Senior-level', 'Manager'], key='experience_level')

    if st.button('Predict Interest'):
        if title and description and location and company_name:
            features = pd.DataFrame({
                'title': [title],
                'description': [description],
                'location': [location],
                'company_name': [company_name],
                'views': [views],
                'applies': [applies],
                'average_salary': [average_salary],
                'work_type': [work_type],
                'experience_level': [experience_level]
            })
            prediction = predictor_model.predict(features)
            st.write('Predicted Candidate Interest:', 'High' if prediction[0] == 1 else 'Low')
        else:
            st.error('Please fill in all fields.')

# Footer
st.write('Made with ❤️ by Your Name')
