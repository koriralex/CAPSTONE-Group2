import pickle
import streamlit as st

# Function to load a single model
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

try:
    description_model = load_model('description.pkl')
    st.write("Loaded description model successfully.")
except Exception as e:
    st.write(f"Error loading description model: {e}")

try:
    knn_model = load_model('knn.pkl')
    st.write("Loaded KNN model successfully.")
except Exception as e:
    st.write(f"Error loading KNN model: {e}")

try:
    predictor_model = load_model('predictor.pkl')
    st.write("Loaded predictor model successfully.")
except Exception as e:
    st.write(f"Error loading predictor model: {e}")
