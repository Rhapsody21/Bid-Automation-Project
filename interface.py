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
            model="gpt-4o-mini" ,           messages=[
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
    doc = fitz.open(pdf_path)  # Use the file path here
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    return text

uploaded_file = st.file_uploader("Upload RFP file", type=["pdf"])
def search_similar_technical_proposal(query):
    """
    Search for similar technical proposals and return the COMPLETE document
    by finding all chunks belonging to the best matching document.
    """
    # Create embedding for the search query
    query_embedding = embeddings.embed_query(query)
    
    # First query to find the best matching document
    initial_results = index.query(
        vector=query_embedding,
        filter={"collection_type": "technical_proposal"},
        top_k=10,
        include_metadata=True
    )

    if not initial_results or not initial_results.get("matches") or len(initial_results["matches"]) == 0:
        return "No technical proposals found in the database."

    # Group matches by document and find best document
    document_scores = {}
    
    for match in initial_results["matches"]:
        metadata = match.get("metadata", {})
        filename = metadata.get("filename", "unknown")
        similarity_score = match.get("score", 0)
        
        # Track highest score per document
        if filename not in document_scores or similarity_score > document_scores[filename]:
            document_scores[filename] = similarity_score
    
    # Find the best document
    best_document = None
    best_score = 0
    
    for filename, score in document_scores.items():
        if score > best_score:
            best_score = score
            best_document = filename
    
    # Set minimum acceptable score threshold
    min_similarity_threshold = 0.70  # Adjust based on testing
    
    if not best_document or best_score < min_similarity_threshold:
        # Prepare alternative suggestions
        alternatives = "\n\n**Top Alternatives:**\n"
        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (doc, score) in enumerate(sorted_docs[:3], 1):
            alternatives += f"{i}. **{doc}** (Score: {score:.2f})\n"
            
        return (f"No sufficiently similar technical proposal found (threshold: {min_similarity_threshold})." + 
                alternatives +
                "\nConsider adjusting search terms or manually reviewing these alternatives.")
    
    # Found a good match, now retrieve ALL chunks for this document
    all_chunks_query = index.query(
        vector=query_embedding,  # Using same embedding for consistency
        filter={
            "collection_type": "technical_proposal",
            "filename": best_document
        },
        top_k=100,  # Set high to get all chunks
        include_metadata=True
    )
    
    # Extract all chunks and organize them by chunk_id
    document_chunks = []
    for chunk_match in all_chunks_query.get("matches", []):
        metadata = chunk_match.get("metadata", {})
        chunk_id = metadata.get("chunk_id", 0)
        text = metadata.get("text", "")
        
        document_chunks.append({
            "chunk_id": chunk_id,
            "text": text
        })
    
    # Sort chunks by chunk_id to reconstruct the document in correct order
    document_chunks.sort(key=lambda x: x["chunk_id"])
    
    # Combine all chunks into a complete document
    complete_document_text = "\n\n".join([chunk["text"] for chunk in document_chunks])
    
    return (f"**Match Score:** {best_score:.2f}\n"
            f"**Document Name:** {best_document}\n\n"
            f"**Complete Document Content:**\n{complete_document_text}")




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
        st.text_area(key_phrases, height=300)
        st.subheader("Searching for the Most Similar Technical Proposal...")
        similar_proposal = search_similar_technical_proposal(key_phrases)
        st.text_area("Most Similar Technical Proposal", similar_proposal, height=300)
    else:
        st.write("No key phrases extracted.")

