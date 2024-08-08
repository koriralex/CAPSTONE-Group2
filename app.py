import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import os

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

# Add custom CSS for styling from an external GitHub file
st.markdown(
    """
    <style>
    @import url('https://raw.githubusercontent.com/your-username/your-repository/main/styles.css');
    </style>
    """,
    unsafe_allow_html=True
)

# Function to display header
def display_header(title, image_url):
    st.markdown(f'<h1 class="title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(f'<img src="{image_url}" alt="Header Image" style="width: 100%; border-radius: 8px;">', unsafe_allow_html=True)

# Image URL for headers
header_image_url = "https://github.com/user-attachments/assets/e4b4502f-f99e-4dce-ad20-122843029701"

# Sidebar for selecting page
page = st.sidebar.radio('Select Page', ['Profile', 'Job Recommendations', 'Predict Candidate Interest', 'Feedback'])

# Function to display profile in the sidebar
def sidebar_profile():
    st.sidebar.markdown("## User Profile")
    first_name = st.sidebar.text_input("First Name:")
    last_name = st.sidebar.text_input("Last Name:")
    full_name = f"{first_name} {last_name}"

    uploaded_photo = st.sidebar.file_uploader("Upload a profile photo (JPG, PNG):", type=['jpg', 'png'], key="profile_photo")
    
    if st.sidebar.button("Save Profile"):
        if uploaded_photo is not None:
            photo_path = save_uploaded_file(uploaded_photo, full_name)
            st.sidebar.image(photo_path, caption='Profile Photo', use_column_width=True, output_format='JPEG')
            st.sidebar.markdown(f"**Name:** {full_name}")
            return full_name, photo_path
        else:
            st.sidebar.markdown("Please upload a profile photo.")
            return full_name, None
    return full_name, None

username, profile_photo_path = sidebar_profile()

# Profile Page
if page == 'Profile':
    display_header('Update Your Profile', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)

    # Profile photo upload
    uploaded_photo = st.file_uploader("Upload a profile photo (JPG, PNG):", type=['jpg', 'png'], key="profile_page_photo")
    
    if uploaded_photo is not None:
        photo_path = save_uploaded_file(uploaded_photo, username)
        st.image(photo_path, caption='Uploaded Profile Photo', use_column_width=True, output_format='JPEG')
        st.markdown('<div class="success-message">Profile photo uploaded successfully!</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Job Recommendations Page
elif page == 'Job Recommendations':
    display_header('Job Recommendations', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)

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
                recommendations = []
                for desc in descriptions:
                    desc_tfidf = vectorizer.transform([desc])
                    similarity_scores = cosine_similarity(desc_tfidf, tfidf_matrix)
                    top_indices = similarity_scores[0].argsort()[-5:][::-1]
                    recommendations.append(df.iloc[top_indices][['title', 'company_name', 'location']])
                st.write('Recommended Jobs:')
                for rec in recommendations:
                    st.write(rec)
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
                    distances, indices = knn_model.kneighbors([tfidf_matrix[job_id_index]])
                    similar_jobs = df.iloc[indices.flatten()]
                    st.write('Recommended Jobs:')
                    st.write(similar_jobs[['title', 'company_name', 'location']])
                else:
                    st.markdown('<div class="error-message">Invalid Job ID selected.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-message">Please select a job ID.</div>', unsafe_allow_html=True)

    # Job Recommendations Based on Title Filter
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

    uploaded_file = st.file_uploader("Upload a file with candidate data (CSV):", type=['csv'])

    if uploaded_file is not None:
        candidate_df = pd.read_csv(uploaded_file)
        required_columns = ['candidate_id', 'job_id', 'views', 'applies', 'average_salary']
        if all(col in candidate_df.columns for col in required_columns):
            candidate_df['predicted_interest'] = forest_model.predict(candidate_df[['views', 'applies', 'average_salary']])
            st.write('Candidate Interest Predictions:')
            st.write(candidate_df[['candidate_id', 'job_id', 'predicted_interest']])
        else:
            st.markdown(f'<div class="error-message">The uploaded file must contain columns: {", ".join(required_columns)}.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Feedback Page
elif page == 'Feedback':
    display_header('Feedback', header_image_url)
    st.markdown('<div class="container main">', unsafe_allow_html=True)
    st.subheader('Provide Your Feedback', anchor='subtitle')

    feedback = st.text_area("Your feedback:")
    if st.button('Submit Feedback'):
        st.markdown('<div class="success-message">Thank you for your feedback!</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
