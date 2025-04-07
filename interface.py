import streamlit as st
import os
import openai
import fitz  # PyMuPDF
import tempfile
from dotenv import load_dotenv
import re
import random
import string

# Import required libraries
from pinecone import Pinecone as PineconeClient
from langchain_openai import OpenAIEmbeddings
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Configuration and API Key Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "bid-project-index"

# Initialize Clients
pc_client = PineconeClient(api_key=PINECONE_API_KEY)
index = pc_client.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Streamlit Page Configuration
st.set_page_config(page_title="RFP Proposal Generator", layout="wide")

class ProposalGenerator:
    @staticmethod
    def fullscreen_text_area(text, height=300, key=None):
        """
        Create a fullscreen text viewer with a floating button.
        
        Args:
            text (str): Text to display
            height (int): Initial height of text area
            key (str, optional): Unique key for the component
        
        Returns:
            str: Displayed text
        """
        if key is None:
            key = ''.join(random.choices(string.ascii_lowercase, k=10))
        
        viewer_id = f"doc-viewer-{key}"
        btn_id = f"floating-fs-btn-{key}"
        
        # JavaScript for Full-Screen Mode
        fullscreen_js = f"""
        <script>
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
        </script>
        """
        
        # Text Area and Floating Button
        text_area = st.text_area(f"Content", text, height=height, key=key)
        
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
        
        st.components.v1.html(html, height=0)
        
        return text_area

    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        
        Returns:
            str: Extracted text from PDF
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return ""

    @staticmethod
    def extract_key_words(rfp_content):
        """
        Extract key words from RFP content using OpenAI.
        
        Args:
            rfp_content (str): RFP document content
        
        Returns:
            str: Extracted key phrases
        """
        try:
            openai.api_key = OPENAI_API_KEY

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert RFP analysis assistant. Extract critical keywords and phrases that define the proposal's core requirements."
                    },
                    {
                        "role": "user",
                        "content": rfp_content
                    }
                ],
                max_tokens=500
            )

            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error extracting key words: {e}")
            return None

    @staticmethod
    def search_similar_proposals(query):
        """
        Search for similar technical proposals.
        
        Args:
            query (str): Search query
        
        Returns:
            list: Top matching proposals
        """
        query_embedding = embeddings.embed_query(query)
        
        results = index.query(
            vector=query_embedding,
            filter={"collection_type": "technical_proposal"},
            top_k=3,
            include_metadata=True
        )

        if not results or not results.get("matches"):
            return "No similar proposals found."

        matched_proposals = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            score = match.get("score", 0)
            
            # Fetch full document chunks
            chunks_query = index.query(
                vector=query_embedding,
                filter={"collection_type": "technical_proposal", "filename": filename},
                top_k=50,
                include_metadata=True
            )
            
            document_chunks = sorted(
                [{"chunk_id": chunk.get("metadata", {}).get("chunk_id", 0), 
                  "text": chunk.get("metadata", {}).get("text", "")} 
                 for chunk in chunks_query.get("matches", [])],
                key=lambda x: x["chunk_id"]
            )
            
            complete_text = "\n\n".join([chunk["text"] for chunk in document_chunks])
            matched_proposals.append((filename, score, complete_text))
        
        return matched_proposals

    @staticmethod
    def generate_proposal(rfp_content, key_phrases, similar_proposals):
        """
        Generate a comprehensive proposal using OpenAI.
        
        Args:
            rfp_content (str): RFP document content
            key_phrases (str): Extracted key phrases
            similar_proposals (list): Similar proposal documents
        
        Returns:
            str: Generated proposal
        """
        try:
            openai.api_key = OPENAI_API_KEY

            # Prepare reference proposals
            reference_content = "\n\n".join([
                f"EXAMPLE PROPOSAL (Score: {score:.2f}):\n{content[:5000]}"
                for _, score, content in similar_proposals[:3]
            ])

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior proposal writer specializing in crafting winning technical proposals."
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Generate a comprehensive technical proposal methodology based on:
                        
                        RFP KEY REQUIREMENTS:
                        {key_phrases}
                        
                        REFERENCE PROPOSALS:
                        {reference_content}
                        
                        Guidelines:
                        - Create a detailed, structured methodology statement
                        - Use professional, technical language
                        - Demonstrate deep understanding of project requirements
                        - Incorporate best practices from reference proposals
                        """
                    }
                ],
                max_tokens=16000,
                temperature=0.7
            )

            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Proposal generation error: {e}")
            return None

    @staticmethod
    def extract_headings(text):
        """
        Extract markdown headings from text.
        
        Args:
            text (str): Input text
        
        Returns:
            list: List of headings
        """
        pattern = r"^(#{1,4}\s.*?)$"
        return re.findall(pattern, text, re.MULTILINE)
    
    @staticmethod
    def extract_section_content(generated_proposal, selected_section):
        """
        Extract the content of a specific section from the generated proposal.
        
        Args:
            generated_proposal (str): Full generated proposal text
            selected_section (str): Section heading to extract
        
        Returns:
            str: Content of the specified section
        """
        # Escape special regex characters in the section name
        escaped_section = re.escape(selected_section.strip())
        
        # Regex pattern to find section content
        # This will capture all text from the section heading to the next heading or end of document
        pattern = rf"{escaped_section}(.*?)(?=\n#|\Z)"
        
        # Search for the section content
        match = re.search(pattern, generated_proposal, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        else:
            return "Section not found in the proposal."

    @staticmethod
    def expand_section(section_content):
        """
        Expand a section using OpenAI to provide more detailed explanation.
        
        Args:
            section_content (str): Content of the section to expand
        
        Returns:
            str: Expanded and detailed explanation of the section
        """
        try:
            openai.api_key = OPENAI_API_KEY

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical writer specializing in providing comprehensive, detailed explanations of proposal sections."
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Provide an extremely detailed, comprehensive expansion of the following section. 
                        Go beyond the original text and offer:
                        - In-depth context
                        - Strategic insights
                        - Comprehensive rationale
                        - Potential implementation approaches
                        - Detailed explanations of key concepts
                        - Potential challenges and mitigation strategies

                        Original Section Content:
                        {section_content}

                        Expansion Guidelines:
                        - Use professional, technical language
                        - Provide clear, structured explanations
                        - Add significant depth and context
                        - Demonstrate expert-level understanding
                        """
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error in section expansion: {str(e)}"

def main():
    st.title("Technical Proposal Generator")

    # Initialize session state variables if they don't exist
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'rfp_text' not in st.session_state:
        st.session_state.rfp_text = None
    if 'key_phrases' not in st.session_state:
        st.session_state.key_phrases = None
    if 'similar_proposals' not in st.session_state:
        st.session_state.similar_proposals = None
    if 'generated_proposal' not in st.session_state:
        st.session_state.generated_proposal = None
    if 'sections' not in st.session_state:
        st.session_state.sections = []
    if 'section_content' not in st.session_state:
        st.session_state.section_content = None
    if 'expanded_section' not in st.session_state:
        st.session_state.expanded_section = None
    if 'displayed_expanded_content' not in st.session_state:
        st.session_state.displayed_expanded_content = False
    if 'displayed_section_content' not in st.session_state:
        st.session_state.displayed_section_content = False
    if 'selected_section_title' not in st.session_state:
        st.session_state.selected_section_title = None

    # Callbacks for buttons to avoid rerunning on state change
    def generate_proposal_callback():
        st.session_state.generate_proposal_clicked = True
    
    def view_section_callback():
        st.session_state.view_section_clicked = True
    
    def expand_section_callback():
        st.session_state.expand_section_clicked = True

    # File uploader
    uploaded_file = st.file_uploader("Upload RFP Document", type=["pdf"])
    
    # Process new file upload
    if uploaded_file and (st.session_state.uploaded_file is None or uploaded_file.name != st.session_state.uploaded_file.name):
        st.session_state.uploaded_file = uploaded_file
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.success("File uploaded successfully")

        # Extract and display RFP content
        rfp_text = ProposalGenerator.extract_text_from_pdf(temp_file_path)
        st.session_state.rfp_text = rfp_text
        
        # Reset related state variables on new file upload
        st.session_state.key_phrases = None
        st.session_state.similar_proposals = None
        st.session_state.generated_proposal = None
        st.session_state.sections = []
        st.session_state.section_content = None
        st.session_state.expanded_section = None
        st.session_state.displayed_expanded_content = False
        st.session_state.displayed_section_content = False
        
    # Display RFP content if available
    if st.session_state.rfp_text:
        st.subheader("RFP Content")
        with st.expander("View RFP content", expanded=False):
            ProposalGenerator.fullscreen_text_area(st.session_state.rfp_text, height=300, key="rfp_content")
        
        # Extract key phrases if not already done
        if st.session_state.key_phrases is None:
            with st.spinner("Analyzing RFP..."):
                key_phrases = ProposalGenerator.extract_key_words(st.session_state.rfp_text)
                st.session_state.key_phrases = key_phrases
        
        # Display key phrases if available
        if st.session_state.key_phrases:
            st.subheader("Extracted Key Requirements")
            with st.expander("View extracted key requirements", expanded=False):
                ProposalGenerator.fullscreen_text_area(st.session_state.key_phrases, height=200, key="key_phrases")
            
            # Search for similar proposals if not already done
            if st.session_state.similar_proposals is None:
                with st.spinner("Finding similar proposals..."):
                    similar_proposals = ProposalGenerator.search_similar_proposals(st.session_state.key_phrases)
                    st.session_state.similar_proposals = similar_proposals
            
            # Display similar proposals if available
            if isinstance(st.session_state.similar_proposals, list):
                st.subheader("Similar Technical Proposals")
                for idx, (filename, score, content) in enumerate(st.session_state.similar_proposals, 1):
                    with st.expander(f"Proposal {idx}: {filename} (Similarity: {score:.2f})", expanded=False):
                        ProposalGenerator.fullscreen_text_area(content, height=500, key=f"proposal_{idx}")
                
                # Generate proposal button
                if st.button("Generate Proposal", on_click=generate_proposal_callback):
                    st.session_state.generate_proposal_clicked = True
                
                # Handle proposal generation
                if 'generate_proposal_clicked' in st.session_state and st.session_state.generate_proposal_clicked:
                    if st.session_state.generated_proposal is None:
                        with st.spinner("Generating comprehensive proposal..."):
                            generated_proposal = ProposalGenerator.generate_proposal(
                                st.session_state.rfp_text, 
                                st.session_state.key_phrases, 
                                st.session_state.similar_proposals
                            )
                            st.session_state.generated_proposal = generated_proposal
                            st.session_state.sections = ProposalGenerator.extract_headings(generated_proposal)
                    
                    # Reset the flag
                    st.session_state.generate_proposal_clicked = False
    
    # Display generated proposal and section tools if available
    if st.session_state.generated_proposal:
        st.subheader("Generated Proposal")
        with st.expander("View full generated proposal", expanded=False):
            ProposalGenerator.fullscreen_text_area(st.session_state.generated_proposal, height=600, key="generated_proposal")
        
        # Download button
        st.download_button(
            label="Download Proposal",
            data=st.session_state.generated_proposal,
            file_name="technical_proposal.txt",
            mime="text/plain"
        )
        
        # Section selection and operations
        st.subheader("Proposal Sections")
        
        # Section selector dropdown
        selected_section = st.selectbox(
            "Choose a section:", 
            st.session_state.sections,
            key="section_selector"
        )
        st.session_state.selected_section_title = selected_section
        
        # Buttons for section operations in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("View Selected Section", on_click=view_section_callback):
                pass
        
        with col2:
            if st.button("Expand Selected Section", on_click=expand_section_callback):
                pass
        
        # Handle view section button
        if 'view_section_clicked' in st.session_state and st.session_state.view_section_clicked:
            with st.spinner(f"Extracting section content..."):
                section_content = ProposalGenerator.extract_section_content(
                    st.session_state.generated_proposal, 
                    st.session_state.selected_section_title
                )
                st.session_state.section_content = section_content
                st.session_state.displayed_section_content = True
                st.session_state.displayed_expanded_content = False
            
            # Reset the flag
            st.session_state.view_section_clicked = False
        
        # Handle expand section button
        if 'expand_section_clicked' in st.session_state and st.session_state.expand_section_clicked:
            with st.spinner(f"Expanding section..."):
                base_content = ProposalGenerator.extract_section_content(
                    st.session_state.generated_proposal, 
                    st.session_state.selected_section_title
                )
                expanded_content = ProposalGenerator.expand_section(base_content)
                st.session_state.expanded_section = expanded_content
                st.session_state.displayed_expanded_content = True
                st.session_state.displayed_section_content = False
            
            # Reset the flag
            st.session_state.expand_section_clicked = False
        
        # Display the content based on what was selected
        if st.session_state.displayed_section_content and st.session_state.section_content:
            st.subheader(f"Section Content: {st.session_state.selected_section_title}")
            st.markdown(st.session_state.section_content)
        
        if st.session_state.displayed_expanded_content and st.session_state.expanded_section:
            st.subheader(f"Expanded Section: {st.session_state.selected_section_title}")
            st.markdown(st.session_state.expanded_section)

if __name__ == "__main__":
    main()