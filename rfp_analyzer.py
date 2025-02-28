import os
from openai import OpenAI
import fitz

def extract_key_words(rfp_content):
    try:
        api_key=os.environ("OPENAI_API_KEY")
    except:
        print("please make sure you provide a valid OPENAI_API_KEY by setting it up in the environment")
    openAiClient=OpenAI(api_key=api_key)
    response= openAiClient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant specialized in analyzing Request for Proposals (RFPs). Your primary task is to extract the most important keywords and phrases that help in preparing a strong proposal. Analyze the RFP document and extract key information, focusing on the most critical sections, including Scope of services (tasks, deliverables, project goals), Eligibility Criteria (who is qualified to bid), Evaluation Criteria (how proposals will be assessed), Required Documents (mandatory attachments), and Compliance & Legal Requirements (regulations, security, ownership rights). Extract full key phrases instead of single words, maintain accuracy by only extracting what is explicitly stated in the RFP, and ignore common words such as greetings and general instructions. If a section is missing, return nothing instead of making assumptions. The output should be structured clearly to help users quickly identify the key details needed for proposal preparation."
            },
            {
                "role": "user",
                "content": rfp_content
            }
        ]
    )
    return response.choices[0].message
    
def extract_text_from_pdf(pdf_path):
    doc=fitz.open(pdf_path)
    text= ""
    for page in doc:
        text +=page.get_text("text") +"\n\n"
    return text
