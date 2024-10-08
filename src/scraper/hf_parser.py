from transformers import pipeline
from bs4 import BeautifulSoup
import streamlit as st

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # st.write(soup.get_text(separator=' ', strip=True))
    return soup.get_text(separator=' ', strip=True)

def extract_job_info(html_content, url):
    # Clean the HTML content
    cleaned_text = clean_html(html_content)
    
    # Initialize the question-answering pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    # Define questions to extract information
    questions = [
        "What is the job title?",
        "What is the name of the company?",
        "What is the location of the job?",
        "What is the salary Range?",
        "What are the main responsibilities of the job?",
        "What are the required qualifications for the job?",
        "What are the benefits offered for this job?",
    ]
    
    # Extract information
    job_info = {}
    for question in questions:
        result = qa_pipeline(question=question, context=cleaned_text[:10000])  # Limit context to 512 tokens
        job_info[question] = result['answer']
    
    # Format the extracted information
    formatted_info = f"Job Information extracted from {url}:\n\n"
    for question, answer in job_info.items():
        formatted_info += f"{question}\n{answer}\n\n"
    
    return formatted_info

def parse_job_info(job_info):
    # Parse the job information into a structured format
    structured_info = {}
    lines = job_info.split('\n')
    current_key = None

    for line in lines:
        if line.endswith('?'):
            current_key = line.strip('?')
            structured_info[current_key] = ""
        elif current_key and line.strip():
            structured_info[current_key] += line.strip() + " "

    # Clean up any trailing spaces
    for key in structured_info:
        structured_info[key] = structured_info[key].strip()

    return structured_info