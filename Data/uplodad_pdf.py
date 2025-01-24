import os
from langchain.document_loaders import PyPDFLoader
import weaviate

# Step 1: Recursively Find All PDF Files in a Directory
def find_all_pdfs(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


# Step 2: Set up Weaviate Schema
def setup_schema(client, class_name):
    schema = {
        "classes": [
            {
                "class": class_name,
                "description": "A class to store document content",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "The main text content of the document",
                    },
                    {
                        "name": "file_name",
                        "dataType": ["text"],
                        "description": "The name of the file",
                    },
                ],
            }
        ]
    }
    client.schema.delete_all()  # Clear any existing schema
    client.schema.create(schema)


# Step 3: Upload PDFs to Weaviate
def upload_pdfs_to_weaviate(pdf_directory, weaviate_url, class_name):
    # Find all PDFs in the directory and subdirectories
    pdf_files = find_all_pdfs(pdf_directory)
    print(f"Found {len(pdf_files)} PDF files.")

    # Connect to Weaviate
    client = weaviate.Client(url=weaviate_url)

    # Ensure the schema is set up
    setup_schema(client, class_name)

    # Process and upload each PDF
    for pdf_file in pdf_files:
        try:
            # Load PDF content
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()

            # Prepare data for Weaviate
            for doc in documents:
                data_object = {
                    "content": doc.page_content,
                    "file_name": os.path.basename(pdf_file),
                }
                client.data_object.create(data_object, class_name)
            print(f"Uploaded: {pdf_file}")
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")


# Main Script
if __name__ == "__main__":
    # Directory containing PDFs (including subdirectories)
    pdf_directory = "./Technical Proposals.pdfs/"  # Replace with the root folder path

    # Weaviate settings
    weaviate_url = "http://localhost:8080"  # Update with your Weaviate instance URL
    class_name = "Document"  # Class name in Weaviate

    # Upload all PDFs
    upload_pdfs_to_weaviate(pdf_directory, weaviate_url, class_name)
