import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import re
import os

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
        st.error(f"Error loading forest model: {e}")
        forest_model = None

    return description_model, knn_model, forest_model

description_model, knn_model, forest_model = load_models()

# Load dataset for KNN recommendations
df = pd.read_csv('postings.csv')

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

# Load job titles and IDs for dropdowns
try:
    title_df = pd.read_csv('title_list.csv')
    job_titles = title_df['title'].tolist()
except Exception as e:
    st.error(f"Error loading job titles: {e}")
    job_titles = []

try:
    job_id_df = pd.read_csv('job_id_list.csv')
    job_ids = job_id_df['job_id'].tolist()
except Exception as e:
    st.error(f"Error loading job IDs: {e}")
    job_ids = []

# Set up the Streamlit app
st.set_page_config(page_title="Job Recommendation System", page_icon="📈", layout="wide")

# Add custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f2f6;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 1200px;
        margin: auto;
        padding: 2rem;
    }
    .title {
        font-size: 2rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .main {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .upload-button, .recommend-button {
        background-color: #007bff;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        font-size: 1rem;
        margin-top: 1rem;
    }
    .upload-button:hover, .recommend-button:hover {
        background-color: #0056b3;
    }
    .uploaded-photo {
        margin-top: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .success-message, .error-message {
        font-size: 1rem;
        margin-top: 1rem;
    }
    .success-message {
        color: #28a745;
    }
    .error-message {
        color: #dc3545;
    }
    .page-content {
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up the Streamlit app layout
st.title('Job Recommendation System', anchor='title')

# Sidebar for selecting page
page = st.sidebar.selectbox('Select Page', ['Profile Update', 'Job Recommendations', 'Predict Candidate Interest', 'Feedback'])

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to recommend jobs based on description
def recommend_jobs(input_description, top_n=10):
    input_description_processed = preprocess_text(input_description)
    input_vector = vectorizer.transform([input_description_processed])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[indices]

# Profile Update Page
if page == 'Profile Update':
    st.markdown('<div class="container main">', unsafe_allow_html=True)
    st.subheader('Update Your Profile', anchor='subtitle')

    # Profile photo upload
    uploaded_photo = st.file_uploader("Upload a profile photo (JPG, PNG):", type=['jpg', 'png'])
    
    if uploaded_photo is not None:
        # Example user ID
        user_id = 'example_user_id'
        photo_path = save_uploaded_file(uploaded_photo, user_id)
        st.image(photo_path, caption='Uploaded Profile Photo', use_column_width=True, output_format='JPEG', class_='uploaded-photo')
        st.markdown('<div class="success-message">Profile photo uploaded successfully!</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Job Recommendations Page
elif page == 'Job Recommendations':
    st.markdown('<div class="container main">', unsafe_allow_html=True)
    st.subheader('Job Recommendations', anchor='subtitle')

    option = st.selectbox('Select Recommendation Type', ['Recommend Jobs Based on Description', 'Recommend Jobs Based on Job ID', 'Recommend Jobs Based on Title Filter'])

    # Job Recommendations Based on Description
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
                    st.markdown('<div class="error-message">Invalid Job ID. Please select a valid ID.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">Please select a job ID.</div>', unsafe_allow_html
                            = True)

    # Job Recommendations Based on Title Filter
    elif option == 'Recommend Jobs Based on Title Filter':
        st.subheader('Job Recommendations Based on Title Filter')
        selected_job_title = st.selectbox('Select Job Title:', options=job_titles)
        
        if st.button('Get Recommendations'):
            if selected_job_title is not None:
                filtered_df = df[df['title'].str.contains(selected_job_title, case=False, na=False)]
                st.write('Recommended Jobs:')
                st.write(filtered_df[['title', 'company_name', 'location']])
            else:
                st.markdown('<div class="error-message">Please select a job title.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Predict Candidate Interest Page
elif page == 'Predict Candidate Interest':
    st.markdown('<div class="container main">', unsafe_allow_html=True)
    st.subheader('Predict Candidate Interest', anchor='subtitle')

    uploaded_file = st.file_uploader("Upload a file with candidate and job details (CSV):", type=['csv'])

    if uploaded_file is not None:
        file_df = pd.read_csv(uploaded_file)
        
        if all(col in file_df.columns for col in ['candidate_id', 'job_id', 'views', 'applies']):
            predictions = forest_model.predict(file_df[['views', 'applies']])
            file_df['interest'] = predictions
            st.write('Predictions:')
            st.write(file_df[['candidate_id', 'job_id', 'interest']])
        else:
            st.markdown('<div class="error-message">The uploaded file must contain "candidate_id", "job_id", "views", and "applies" columns.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Feedback Page
elif page == 'Feedback':
    st.markdown('<div class="container main">', unsafe_allow_html=True)
    st.subheader('Feedback', anchor='subtitle')

    feedback_text = st.text_area("Please provide your feedback:")
    submit_button = st.button("Submit")

    if submit_button and feedback_text:
        st.markdown('<div class="success-message">Thank you for your feedback!</div>', unsafe_allow_html=True)
    elif submit_button:
        st.markdown('<div class="error-message">Feedback cannot be empty.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

