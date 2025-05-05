# import os
# import boto3
# import openai
# import fitz  # PyMuPDF
# from io import BytesIO
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # AWS S3 Configuration
# AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
# AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
# AWS_REGION = "us-east-1"
# S3_BUCKET_NAME = "bid-automation-model-data"
# S3_PREFIXES = ["technical-proposals/", "financial-proposals/"]

# # OpenAI
# openai.api_key = os.environ["OPENAI_API_KEY"]
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# # Initialize S3 Client
# s3_client = boto3.client(
#     "s3",
#     aws_access_key_id=AWS_ACCESS_KEY,
#     aws_secret_access_key=AWS_SECRET_KEY,
#     region_name=AWS_REGION,
# )

# documents = []
# for prefix in S3_PREFIXES:
#     continuation_token = None
#     while True:
#         kwargs = {"Bucket": S3_BUCKET_NAME, "Prefix": prefix}
#         if continuation_token:
#             kwargs["ContinuationToken"] = continuation_token

#         response = s3_client.list_objects_v2(**kwargs)
#         pdf_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".pdf")]

#         for pdf_file in pdf_files:
#             pdf_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=pdf_file)
#             pdf_content = pdf_obj["Body"].read()

#             # ✅ Use PyMuPDF to extract text
#             with fitz.open(stream=pdf_content, filetype="pdf") as doc:
#                 text = "\n".join([page.get_text() for page in doc])

#             if text.strip():
#                 doc_type = "technical" if "technical" in pdf_file.lower() else "financial"
#                 documents.append(Document(
#                     page_content=text,
#                     metadata={"source": pdf_file, "doc_type": doc_type}
#                 ))

#         if response.get("IsTruncated"):
#             continuation_token = response.get("NextContinuationToken")
#         else:
#             break

# # Chunk documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1536, chunk_overlap=50)
# split_documents = []

# for doc in documents:
#     chunks = text_splitter.split_text(doc.page_content)
#     for i, chunk in enumerate(chunks):
#         split_documents.append(Document(
#             page_content=chunk,
#             metadata={
#                 "source": doc.metadata["source"],
#                 "doc_type": doc.metadata["doc_type"],
#                 "chunk_index": i,
#                 "text": chunk
#             }
#         ))

# # Upsert to Pinecone
# pinecone_index_name = "bid-automation-index"
# vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
# vectorstore.add_documents(documents=split_documents)

# print("Embeddings from all S3 PDF files (across multiple folders) created and inserted into Pinecone successfully!")




import os
import boto3
import openai
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# AWS S3 Configuration
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "bid-automation-model-data"
S3_PREFIXES = ["technical-proposals/", "financial-proposals/"]

# OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

# 🧠 Function: Extract text from PDFs (text or image-based)
def extract_text_with_ocr_fallback(pdf_content, pdf_file):
    text = ""
    try:
        with fitz.open(stream=pdf_content, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
                else:
                    # Perform OCR on image-based page
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    text += ocr_text + "\n"
    except Exception as e:
        print(f"❌ Error reading PDF {pdf_file}: {e}")
    
    if not text.strip():
        print(f"⚠️ No extractable or OCR text found in {pdf_file}")
    return text

# 🔍 Extract documents from S3
documents = []
for prefix in S3_PREFIXES:
    continuation_token = None
    while True:
        kwargs = {"Bucket": S3_BUCKET_NAME, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**kwargs)
        pdf_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".pdf")]

        for pdf_file in pdf_files:
            pdf_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=pdf_file)
            pdf_content = pdf_obj["Body"].read()

            # ✅ Extract text (OCR fallback for image-based)
            text = extract_text_with_ocr_fallback(pdf_content, pdf_file)

            if text.strip():
                doc_type = "technical" if "technical" in pdf_file.lower() else "financial"
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf_file, "doc_type": doc_type}
                ))

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

# ✂️ Chunk documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1536, chunk_overlap=50)
split_documents = []

for doc in documents:
    chunks = text_splitter.split_text(doc.page_content)
    for i, chunk in enumerate(chunks):
        split_documents.append(Document(
            page_content=chunk,
            metadata={
                "source": doc.metadata["source"],
                "doc_type": doc.metadata["doc_type"],
                "chunk_index": i,
                "text": chunk
            }
        ))

# 📌 Upsert to Pinecone
pinecone_index_name = "bid-automation-index"
vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
vectorstore.add_documents(documents=split_documents)

print("✅ Embeddings from all S3 PDF files (with OCR support) created and inserted into Pinecone successfully!")