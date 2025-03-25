import streamlit as st
import os
import openai
import fitz
import tempfile
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone as PineconeClient
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from docx import Document
import io
import re


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "bid-project-index"
pc_client = PineconeClient(api_key=PINECONE_API_KEY)
index = pc_client.Index(INDEX_NAME)
st.set_page_config(page_title="My App", layout="wide")

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

st.title("Requirements Extractor")

# Function to create a fullscreen text viewer with floating button
def fullscreen_text_area(text, height=300, key=None):
    # Generate unique keys if none provided
    if key is None:
        import random
        import string
        key = ''.join(random.choices(string.ascii_lowercase, k=10))
    
    viewer_id = f"doc-viewer-{key}"
    btn_id = f"floating-fs-btn-{key}"
    
    # JavaScript for Full-Screen Mode with floating button
    fullscreen_js = f"""
    <script>
        // Function to enter fullscreen
        function openFullScreen_{key}() {{
            var docViewer = document.getElementById("{viewer_id}");
            if (docViewer.requestFullscreen) {{
                docViewer.requestFullscreen();
            }} else if (docViewer.mozRequestFullScreen) {{
                docViewer.mozRequestFullScreen();
            }} else if (docViewer.webkitRequestFullscreen) {{
                docViewer.webkitRequestFullscreen();
            }} else if (docViewer.msRequestFullscreen) {{
                docViewer.msRequestFullscreen();
            }}
        }}

        // Show/hide button when entering/exiting fullscreen
        document.addEventListener("fullscreenchange", function() {{
            if (!document.fullscreenElement) {{
                document.getElementById("{btn_id}").style.display = "flex";
            }} else {{
                document.getElementById("{btn_id}").style.display = "none";
            }}
        }});
    </script>
    """
    
    # First create the regular text area
    text_area = st.text_area(f"Content", text, height=height, key=key)
    
    # Now create the HTML structure with floating button for the same content
    html = f"""
    <div style="position: relative; width: 100%; margin-top: -45px; z-index: 1000;">
        <div id="{viewer_id}" style="position: relative; width: 100%;">
            <pre style="white-space: pre-wrap; word-wrap: break-word; display:none;">{text}</pre>
        </div>
        
        <button id="{btn_id}" onclick="openFullScreen_{key}()" style="
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: rgba(0, 123, 255, 0.7);
            color: white;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 1000;
            transition: all 0.3s ease;
        ">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
            </svg>
        </button>
    </div>
    {fullscreen_js}
    """
    
    # Inject the HTML
    st.components.v1.html(html, height=0)
    
    return text_area

# Function to extract key words using OpenAI API
def extract_key_words(rfp_content):
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API Key is missing")

        openai.api_key = api_key

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": ("You are an AI assistant specialized in analyzing Request for Proposals (RFPs). "
                                "Your primary task is to extract the most important keywords and phrases that help in "
                                "preparing a strong proposal. Analyze the RFP document and extract key information, "
                                "focusing on the most critical sections, including Scope of services (tasks, deliverables, "
                                "project goals), Eligibility Criteria (who is qualified to bid), Evaluation Criteria (how proposals "
                                "will be assessed), Required Documents (mandatory attachments), and Compliance & Legal Requirements "
                                "(regulations, security, ownership rights). Extract full key phrases instead of single words, maintain accuracy "
                                "by only extracting what is explicitly stated in the RFP, and ignore common words such as greetings and general "
                                "instructions. If a section is missing, return nothing instead of making assumptions. The output should be structured "
                                "clearly to help users quickly identify the key details needed for proposal preparation.")
                },
                {
                    "role": "user",
                    "content": rfp_content
                }
            ]
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during OpenAI API request: {e}")
        return None

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    return text


def search_similar_technical_proposal(query):
    """
    Search for similar technical proposals and return the top three best matches
    along with their complete document contents.
    """
    # Create embedding for the search query
    query_embedding = embeddings.embed_query(query)
    
    # Query to find the best matching documents
    initial_results = index.query(
        vector=query_embedding,
        filter={"collection_type": "technical_proposal"},
        top_k=10,
        include_metadata=True
    )

    if not initial_results or not initial_results.get("matches"):
        return "No technical proposals found in the database."

    # Group matches by document and track scores
    document_scores = {}
    
    for match in initial_results["matches"]:
        metadata = match.get("metadata", {})
        filename = metadata.get("filename", "unknown")
        similarity_score = match.get("score", 0)
        
        if filename not in document_scores or similarity_score > document_scores[filename]:
            document_scores[filename] = similarity_score
    
    # Sort documents by score in descending order
    sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if not sorted_docs:
        return "No sufficiently similar technical proposals found."

    # Fetch and display the top three documents
    result_texts = []
    for filename, score in sorted_docs:
        all_chunks_query = index.query(
            vector=query_embedding,
            filter={"collection_type": "technical_proposal", "filename": filename},
            top_k=100,
            include_metadata=True
        )
        
        document_chunks = []
        for chunk_match in all_chunks_query.get("matches", []):
            metadata = chunk_match.get("metadata", {})
            chunk_id = metadata.get("chunk_id", 0)
            text = metadata.get("text", "")
            
            document_chunks.append({"chunk_id": chunk_id, "text": text})
        
        document_chunks.sort(key=lambda x: x["chunk_id"])  # Sort chunks by chunk_id
        complete_document_text = "\n\n".join([chunk["text"] for chunk in document_chunks])
        
        
        result_texts.append((filename, score, complete_document_text))
    
    return result_texts

# def generate_proposal(rfp_content, key_phrases, similar_proposals):
#     """
#     Generate a new proposal based on the RFP content, extracted key phrases, and similar proposals.
    
#     Args:
#         rfp_content (str): The full text of the uploaded RFP
#         key_phrases (str): Extracted key phrases from the RFP
#         similar_proposals (list): List of tuples containing (filename, score, content) for similar proposals
    
#     Returns:
#         str: Generated proposal content
#     """
#     try:
#         api_key = OPENAI_API_KEY
#         if api_key is None:
#             raise ValueError("API Key is missing")

#         openai.api_key = api_key
        
#         # Prepare examples from similar proposals
#         examples_content = ""
#         for idx, (filename, score, content) in enumerate(similar_proposals, start=1):
#             examples_content += f"\n\nEXAMPLE PROPOSAL {idx} (Score: {score:.2f}):\n{content[:10000]}..."  # Limit size of each example
        
#         # Create the prompt
#         prompt = f"""
    
#         KEY REQUIREMENTS EXTRACTED FROM RFP:
#         {key_phrases}
        
#         SIMILAR SUCCESSFUL PROPOSALS FOR REFERENCE:
#         {examples_content}
        
#         you are an AI-powered proposal assistant designed to generate a Methodology Statement for a technical proposal 
#         based on the Scope of Services outlined in extracted requirements {key_phrases} from the rfp and also the example proposals {examples_content}
#         Generate a Structured Methodology Statement
#         - The methodology statement should be detailed and structured, spanning multiple pages as applicable.  
#         - It should be professional, precise, and tailored to the specific consultancy services described in the RFP.  
#         - The content must be clear, technically sound, and demonstrate a comprehensive understanding of the project’s requirements.

#         Methodology Statement Structure
#         1. Introduction
#         - Overview of the consultancy services to be provided.  
#         - Explanation of how the proposed methodology aligns with the client’s needs as stated in the RFP.  
#         - Key objectives of the assignment.

#         2. Scope of the Consultancy Services
#         - Detailed description of the Scope of Services as extracted from the RFP.  
#         - Breakdown of the client’s key expectations, deliverables, and specific tasks required.  
#         - Any technical, regulatory, or operational constraints affecting service execution.

#         3. Project Objectives and Mobilization
#         - Key objectives of the consultancy services.  
#         - Mobilization plan, including initial preparation activities before full-scale implementation.  
#         - Logistics, resources, and preparatory actions to ensure a smooth project start.

#         4. Pre-Commencement Activities
#         (Tailor this section based on the Scope of Services extracted from the RFP)  
#         - Site Inspection & Initial Assessments (if applicable).  
#         - Review & Analysis of Existing Designs (if applicable).  
#         - Collection, Evaluation, and Analysis of Data relevant to the assignment.  
#         - Any preliminary planning activities essential for the consultancy services.  

#         5. Technical Approach and Methodology  
#         (Provide a clear and detailed methodology, adapting to the specific consultancy services required in the RFP.)  
#         5.1 Methodology for Performing the Assignment  
#         - Step-by-step approach detailing how the consultancy services will be executed.  
#         - Use of industry best practices, innovative approaches, and compliance with relevant standards.  
#         - Any specialized methodologies, frameworks, or techniques relevant to the assignment.  
#         5.2 Work Plan for Performing the Assignment 
#         - Detailed breakdown of major tasks, milestones, and timelines.  
#         - Implementation schedule, indicating key phases of the project.  
#         - Integration of monitoring and evaluation mechanisms to track progress.  
#         6. Conclusion
#         - Summary reinforcing the consultant’s approach to successfully delivering the services.  
#         - Commitment to efficiency, quality, and compliance with project goals.  

#         Formatting & Style Guidelines
#         - The response must be customized to the specific consultancy services required.  
#         - Use professional, technical, and persuasive language suitable for a bid submission.  
#         - Maintain logical flow with clear headings, subheadings, and bullet points.  
#         - Ensure industry-standard formatting for professional proposals.  
#         The methodology statement should be comprehensive, structured, and aligned with the client’s expectations as defined in the RFP.
#         """
        
#         # Call the API with a longer max_tokens value and higher temperature for creativity
#         response = openai.chat.completions.create(
#             model="gpt-4o",  # Using full GPT-4o for better quality
#             messages=[
#                 {
#                     "role": "system",
#                     "content": ("You are an elite proposal writer and domain expert with 20+ years of experience winning competitive bids. "
#                                "Your task is to write three specific sections of a proposal that demonstrate superior expertise, "
#                                "innovative thinking, and a deep understanding of the client's needs. "
#                                "Your writing should be confident, technically precise, and compelling. "
#                                "Draw from your extensive experience to create content that stands out from competitors. "
#                                "Be specific, practical, and convincing rather than generic or theoretical. "
#                                "Focus on demonstrating value and addressing the client's pain points throughout your response.")
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             max_tokens=16000,
#             temperature=0.7
#         )
        
#         return response.choices[0].message.content
#     except Exception as e:
#         st.error(f"Error during proposal generation: {e}")
#         return None
def extract_headings(text):

    pattern = r"^(####.*)$"  
    
    # Use re.findall with multiline flag to capture headings
    matches = re.findall(pattern, text, re.MULTILINE)
    
    return matches
def generate_proposal(rfp_content, key_phrases, similar_proposals):
    """
    Generate a new proposal based on the RFP content, extracted key phrases, and similar proposals.
    
    Args:
        rfp_content (str): The full text of the uploaded RFP
        key_phrases (str): Extracted key phrases from the RFP
        similar_proposals (list): List of tuples containing (filename, score, content) for similar proposals
    
    Returns:
        str: Generated proposal content
    """
    try:
        api_key = OPENAI_API_KEY
        if api_key is None:
            raise ValueError("API Key is missing")

        openai.api_key = api_key
        
        # Prepare examples from similar proposals
        examples_content = ""
        for idx, (filename, score, content) in enumerate(similar_proposals, start=1):
            examples_content += f"\n\nEXAMPLE PROPOSAL {idx} (Score: {score:.2f}):\n{content[:10000]}..."  # Limit size of each example
        
        # Create the prompt
        prompt = f"""
    
        KEY REQUIREMENTS EXTRACTED FROM RFP:
        {key_phrases}
        
        SIMILAR SUCCESSFUL PROPOSALS FOR REFERENCE:
        {examples_content}
        
        You are an AI-powered bid proposal assistant designed to generate a detailed, structured, and comprehensive Methodology Statement for a technical proposal.

Your goal is to generate a fully developed Methodology Statement based on:

The Scope of Services extracted from the uploaded Request for Proposal (RFP).

The most relevant past methodology sections retrieved from similar proposals.

Methodology Statement Generation Guidelines:
Use a fully developed paragraph structure rather than summarizing key points.

Incorporate detailed explanations, industry best practices, and project-specific considerations.

Ensure the writing is technical, professional, and persuasive—tailored for bid submission.

Where applicable, reuse content from retrieved methodology statements while adapting them to the new RFP’s requirements.

Structure of the Methodology Statement
1. Introduction
Overview of the consultancy services to be provided.

Explain how the methodology aligns with the client’s requirements.

Key project objectives.

2. Scope of Consultancy Services
Provide a detailed breakdown of the scope of services extracted from the RFP.

Outline key deliverables and tasks required by the client.

3. Project Objectives & Mobilization
Clearly outline the primary objectives of the consultancy services.

Describe the mobilization plan and initial preparation activities.

4. Pre-Commencement Activities (Customized to RFP Requirements)
Site Inspections & Assessments (if applicable).

Review of existing designs, documents, and reports.

Initial data collection, evaluation, and analysis.

5. Technical Approach and Methodology
5.1 Detailed Methodology for Performing the Assignment

Provide a step-by-step breakdown of how the consultancy services will be executed.

Utilize best practices, industry frameworks, and innovative approaches.

Where applicable, explain the rationale behind key methodological choices.

5.2 Work Plan and Execution Strategy

Provide a detailed work plan outlining key phases, milestones, and deliverables.

Include an implementation schedule, showing how the consultancy will progress.

Describe monitoring and evaluation mechanisms to ensure project success.

6. Conclusion
Summarize the consultant’s approach to delivering high-quality services.

Reinforce the firm’s commitment to efficiency, compliance, and client satisfaction.

Key Instructions for Processing
DO NOT SUMMARIZE. Generate a fully developed, detailed, and comprehensive methodology statement.

REUSE relevant content from retrieved methodology sections to enhance the quality and depth of the response.

Ensure detailed paragraph writing, providing a rationale for each methodological choice.

Customize the methodology to the specific project described in the RFP.

Maintain a professional and persuasive tone, suitable for competitive bid submissions.
        """
        
        # Call the API with a longer max_tokens value and higher temperature for creativity
        response = openai.chat.completions.create(
            model="gpt-4o",  # Using full GPT-4o for better quality
            messages=[
                {
                    "role": "system",
                    "content": ("You are an elite proposal writer and domain expert with 20+ years of experience winning competitive bids. "
                               "Your task is to write three specific sections of a proposal that demonstrate superior expertise, "
                               "innovative thinking, and a deep understanding of the client's needs. "
                               "Your writing should be confident, technically precise, and compelling. "
                               "Draw from your extensive experience to create content that stands out from competitors. "
                               "Be specific, practical, and convincing rather than generic or theoretical. "
                               "Focus on demonstrating value and addressing the client's pain points throughout your response.")
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=16000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during proposal generation: {e}")
        return None

def main():
    uploaded_file = st.file_uploader("Upload RFP file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.write("File uploaded successfully")

        # Extract text from the saved PDF file
        rfp_text = extract_text_from_pdf(temp_file_path)

        # Display RFP content with fullscreen capability
        st.write("RFP Content:")
        with st.expander("View RFP Content", expanded=False):
            fullscreen_text_area(rfp_text, height=300, key="rfp_content")

        # Extract key phrases from the RFP content
        with st.spinner("Extracting Key Phrases..."):
            key_phrases = extract_key_words(rfp_text)

        if key_phrases:
            st.write("Extracted Key Phrases:")
            with st.expander("View Extracted Key Phrases", expanded=False):
                fullscreen_text_area(key_phrases, height=300, key="key_phrases")
            
            with st.spinner("Searching for similar proposals..."):
                similar_proposals = search_similar_technical_proposal(key_phrases)
            
            st.subheader("Top Three Most Similar Technical Proposals")
            
            if isinstance(similar_proposals, str):
                st.write(similar_proposals)
                has_similar_proposals = False
            else:
                has_similar_proposals = True
                with st.expander("View Similar Proposals", expanded=False):
                    for idx, (filename, score, content) in enumerate(similar_proposals, start=1):
                        st.subheader(f"Proposal {idx}: {filename} (Score: {score:.2f})")
                        fullscreen_text_area(content, height=500, key=f"proposal_{idx}")
            
            # Add button to generate new proposal
            if has_similar_proposals:
                if st.button("Generate New Proposal"):
                    with st.spinner("Generating proposal based on RFP and similar proposals... This may take a few minutes."):
                        generated_proposal = generate_proposal(rfp_text, key_phrases, similar_proposals)
                    
                    if generated_proposal:
                        st.subheader("Generated Proposal")
                        fullscreen_text_area(generated_proposal, height=500, key="generated_proposal")
                        
                        # Add a download button for the generated proposal
                        st.download_button(
                            label="Download the new Proposal",
                            data=generated_proposal,
                            file_name="generated_proposal.txt",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        sections= extract_headings(generated_proposal)
                        st.write("Select an option from the dropdown")

                           
                        selected_option = st.selectbox("Choose an option:", sections)
                        with st.form(key='my_form'):
                            submitted = st.form_submit_button("Submit")
                        # Display result after submission
                        if submitted:
                            st.success(f"You selected: {selected_option}")
                        else:
                            st.write("No key phrases extracted.")

                        

        else:
            st.write("No key phrases extracted.")

if __name__ == "__main__":
    main()