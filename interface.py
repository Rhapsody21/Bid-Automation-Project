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


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "bid-project-index"
pc_client = PineconeClient(api_key=PINECONE_API_KEY)
index = pc_client.Index(INDEX_NAME)
st.set_page_config(page_title="My App", layout="wide")

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

st.title("RFP Key Phrase Extractor")

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

uploaded_file = st.file_uploader("Upload RFP file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.write("File uploaded successfully")

    # Extract text from the saved PDF file
    rfp_text = extract_text_from_pdf(temp_file_path)

    # Display RFP content
    st.write("RFP Content:")
    st.text_area("RFP Content", rfp_text, height=300)

    # Extract key phrases from the RFP content
    st.write("Extracting Key Phrases...")
    key_phrases = extract_key_words(rfp_text)

    if key_phrases:
        st.write("Extracted Key Phrases:")
        st.text_area("Extracted Key Phrases", key_phrases, height=300)
        
        st.subheader("Top Three Most Similar Technical Proposals")
        similar_proposals = search_similar_technical_proposal(key_phrases)
        
        if isinstance(similar_proposals, str):
            st.write(similar_proposals)
        else:
            for idx, (filename, score, content) in enumerate(similar_proposals, start=1):
                st.subheader(f"Proposal {idx}: {filename} (Score: {score:.2f})")
                st.text_area(f"{filename}", content, height=300)
    else:
        st.write("No key phrases extracted.")