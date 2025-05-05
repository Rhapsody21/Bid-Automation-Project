import streamlit as st
import fitz  # PyMuPDF for PDF reading

# Title of the App
st.title("RFP Requirements Extractor")

# File Uploader
uploaded_file = st.file_uploader("Upload an RFP (PDF)", type=["pdf"])

# Display PDF Content if Uploaded
if uploaded_file is not None:
    st.subheader("Uploaded RFP Content:")
    
    # Load PDF using PyMuPDF
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pdf_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pdf_text += page.get_text()
    
    # Display the extracted text from the PDF
    with st.expander("View Full RFP Text"):
        st.text_area("RFP Content", pdf_text, height=300)
    
    # Placeholder for GPT-4 extraction (Task 2)
    st.subheader("Extracted Requirements:")
    st.info("GPT-4 extraction will be displayed here in the next step.")

# Run using: streamlit run script_name.py
