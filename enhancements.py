import streamlit as st
from streamlit.components.v1 import html

def inject_custom_css():
    st.markdown("""
        <style>
        /* Global dark background and light text */
        body, .stApp {
            background-color: #1e1e1e;  /* Charcoal grey */
            color: #eaeaea;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Main container spacing */
        .main .block-container {
            padding: 2rem 3rem;
        }

        /* Custom header bar */
        .custom-header {
            background-color: #2a2a2a;
            color: #ffffff;
            padding: 1.5rem 2rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            border-left: 5px solid #1f77b4;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #2c2c2c;
            color: #f1f1f1;
        }

        /* Button styling */
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #145a86;
            color: white;
        }

        /* Inputs & textareas */
        textarea, .stTextInput input {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #444;
            border-radius: 6px;
        }

        /* Expander styling */
        .stExpander {
            background-color: #292929;
            color: #e0e0e0;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
        }

        /* Scrollbar tweaks */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1e1e1e;
        }
        ::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
        <div class="custom-header">
            <h1>📂 Bid Proposal Generator</h1>
            <p>Your intelligent assistant for RFP analysis and proposal generation</p>
        </div>
    """, unsafe_allow_html=True)