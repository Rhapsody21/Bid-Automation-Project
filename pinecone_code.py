import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

import openai
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone  # Updated Pinecone import
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

# Function to read PDF files and organize them into a dictionary
def read_documents(directory):
    pdf_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            pdf_dict[filename] = documents  # Store content under the filename

    return pdf_dict

# Function to split documents into chunks
def chunk_doc(doc, chunk_size=1500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(doc)
    return chunks

# Define directories
tech_dir = "./Data/Technical Proposals.pdfs/"
rfps_dir = "./Data/RFPs.pdfs/"
fin_proposals_dir = "./Data/Financial Proposals.pdfs/"

# Read and store PDFs in dictionaries
tech_proposals = read_documents(tech_dir)
rfps_doc = read_documents(rfps_dir)
fin_proposals = read_documents(fin_proposals_dir)

# Chunk the documents
chunked_tech_doc = {name: chunk_doc(doc) for name, doc in tech_proposals.items()}
chunked_rfps_doc = {name: chunk_doc(doc) for name, doc in rfps_doc.items()}
chunked_fin_docs = {name: chunk_doc(doc) for name, doc in fin_proposals.items()}

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

# Pinecone Index Configuration
index_name = "bid-project-index"

# Initialize Pinecone client
pc_client = PineconeClient(api_key=os.environ["PINECONE_API_KEY"])

# Create or connect to the Pinecone index
if index_name not in pc_client.list_indexes():
    pc_client.create_index(
        name=index_name,
        dimension=1536,  # Adjust based on your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the index
index = pc_client.Index(index_name)

# Initialize LangChain Pinecone wrapper
vectorstore = Pinecone(index, embeddings, "text")

# Function to upsert documents into Pinecone with metadata
def upsert_documents(chunked_docs, collection_type):
    for filename, chunks in chunked_docs.items():
        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk
            embedding = embeddings.embed_query(chunk.page_content)
            
            # Prepare metadata
            metadata = {
                "filename": filename,
                "chunk_id": i,
                "collection_type": collection_type,
                "text": chunk.page_content  # Store the actual text for reference
            }
            
            # Upsert into Pinecone
            index.upsert(
                vectors=[{
                    "id": f"{filename}_chunk_{i}",
                    "values": embedding,
                    "metadata": metadata
                }]
            )

# Upsert all documents into Pinecone
upsert_documents(chunked_tech_doc, "technical_proposal")
upsert_documents(chunked_rfps_doc, "rfp")
upsert_documents(chunked_fin_docs, "financial_proposal")

print("All documents have been upserted into Pinecone.")
