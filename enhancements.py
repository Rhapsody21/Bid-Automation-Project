import streamlit as st
from streamlit.components.v1 import html

# Set up page configuration
st.set_page_config(
    page_title="Bid Proposal Generator",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for styling
def inject_custom_css():
    st.markdown("""
        <style>
            body {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', sans-serif;
                color: #212529;
            }

            .stApp {
                background: linear-gradient(to bottom, #ffffff, #f1f3f5);
            }

            h1, h2, h3, h4 {
                color: #2c3e50;
            }

            .stTextArea textarea {
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 0.375rem;
                padding: 0.75rem;
            }

            .stButton button {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 0.375rem;
                padding: 0.5rem 1rem;
                font-size: 1rem;
                transition: background-color 0.3s;
            }

            .stButton button:hover {
                background-color: #0b5ed7;
            }

            .stDownloadButton button {
                background-color: #198754;
                color: white;
                border: none;
                border-radius: 0.375rem;
                padding: 0.5rem 1rem;
                font-size: 1rem;
            }

            .stDownloadButton button:hover {
                background-color: #157347;
            }

            .stSpinner > div > div {
                border-top-color: #0d6efd;
            }
        </style>
    """, unsafe_allow_html=True)


# Optional: Add a header banner or styled title
def render_header():
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='margin-bottom: 0; color: #0d6efd;'>📂 Automated Bid Preparation System (ABPSys)</h1>
            <p style='font-size: 1.1rem; color: #495057;'>Leverage AI to streamline bid proposal generation and enhance response quality</p>
        </div>
    """, unsafe_allow_html=True)


# To be called at the start of your main app
inject_custom_css()
render_header()