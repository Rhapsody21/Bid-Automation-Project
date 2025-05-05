import streamlit as st
import fitz  # PyMuPDF for extracting text from PDFs
import openai
import pinecone
import json

# Initialize Pinecone
pinecone.init(api_key="PINECONE_API_KEY", environment="us-east-1")
index = pinecone.Index("bid-automation-index")

# OpenAI API Key
openai.api_key = "OPENAI_API_KEY"

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_bid_attributes(text):
    """Use OpenAI GPT-4 to extract key bid attributes from the text."""
    prompt = f"""Extract key bid attributes from the following RFP text: {text}. 
    Provide details such as Description of Services, Location, Scope of Services, 
    Eligibility Criteria, Evaluation Criteria, and Required Documentation/Submission Requirements in JSON format."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    
    attributes = json.loads(response["choices"][0]["message"]["content"])
    return attributes

def generate_embeddings(attributes):
    """Convert extracted attributes into embeddings using OpenAI's model."""
    text_to_embed = json.dumps(attributes)
    embedding_response = openai.Embedding.create(
        input=text_to_embed,
        model="text-embedding-ada-002"
    )
    embedding_vector = embedding_response["data"][0]["embedding"]
    return embedding_vector

def store_in_pinecone(doc_id, attributes, embedding):
    """Store extracted bid attributes and embeddings in Pinecone."""
    index.upsert([(doc_id, embedding, attributes)])

def search_similar_bids(query_text):
    """Search for similar past bids in Pinecone."""
    embedding = generate_embeddings({"query": query_text})
    results = index.query(vector=embedding, top_k=3, include_metadata=True)
    return results["matches"]

# Streamlit UI
st.title("Bid Proposal Attribute Extractor")
uploaded_file = st.file_uploader("Upload an RFP PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        
    with st.spinner("Extracting key attributes..."):
        attributes = extract_bid_attributes(text)
    
    with st.spinner("Generating embeddings..."):
        embedding = generate_embeddings(attributes)
    
    with st.spinner("Storing data in Pinecone..."):
        doc_id = uploaded_file.name  # Use filename as document ID
        store_in_pinecone(doc_id, attributes, embedding)
    
    st.success("Extraction & Storage Complete!")
    st.json(attributes)
    
    search_query = st.text_input("Search for similar past bids:")
    if search_query:
        with st.spinner("Searching Pinecone..."):
            results = search_similar_bids(search_query)
        
        st.subheader("Similar Past Bids")
        for match in results:
            st.json(match["metadata"])