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

        section[data-testid="stSidebar"] label {
            font-size: 1rem;
            margin-bottom: 1rem;
        }

        section[data-testid="stSidebar"] .stRadio > div {
            gap: 1rem;
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
            <h1>📂 Bid Preparation System</h1>
            <p>Your intelligent assistant for bid proposal generation</p>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar_menu():
    with st.sidebar:
        st.markdown("""
            <style>
                .sidebar-title {
                    font-size: 26px;
                    font-weight: 700;
                    color: #ffffff;
                    margin-bottom: 0.5rem;
                }
                .sidebar-subtitle {
                    font-size: 15px;
                    color: #cccccc;
                    margin-bottom: 1rem;
                }
                .info-box {
                    background-color: rgba(255, 255, 255, 0.05);
                    padding: 12px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    border-left: 4px solid #3399ff;
                    color: #e0e0e0;
                    font-size: 15px;
                }
                .tip-box {
                    background-color: rgba(102, 187, 106, 0.15);
                    padding: 12px;
                    border-radius: 10px;
                    border-left: 4px solid #66bb6a;
                    color: #e0e0e0;
                    font-size: 14px;
                }
                .info-item {
                    margin-top: 5px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">The ABPSys</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Your AI-powered Bid companion</div>', unsafe_allow_html=True)

        st.markdown("""
            <div class="info-box">
                <strong>What It Does:</strong><br>
                    This tool helps you:
                <div class="info-item">📄 Extract RFP requirements</div>
                <div class="info-item">🔍 Find similar proposals</div>
                <div class="info-item">🧠 Generate tailored methodologies</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="tip-box">
                📌 <strong>Usage Tip:</strong> Upload your RFP on the main screen to get started.<br>
                📂 <strong>Supported Format:</strong> PDF only.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<p style="color: #888888; font-size: 13px;">🛠️ Developed for research and prototyping purposes<br>📅 Version: May 2025</p>', unsafe_allow_html=True)