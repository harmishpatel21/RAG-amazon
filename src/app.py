import streamlit as st
from scraper.job_scraper import fetch_job_posting_html
from scraper.hf_parser import extract_job_info, parse_job_info
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the sentence transformer model
@st.cache(allow_output_mutation=True)
def load_sentence_transformer():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_sentence_transformer()

st.title("Job Posting RAG App")

url = st.text_input("Enter job posting URL:")

if url:
    try:
        html_content = fetch_job_posting_html(url)
        job_info = extract_job_info(html_content, url)
        structured_info = parse_job_info(job_info)
        
        st.subheader("Job Information")
        st.write(structured_info)
        
        # Create embeddings for the job information
        job_embedding = model.encode([structured_info])[0]
        
        query = st.text_input("Ask a question about the job posting:")
        
        if query:
            # Create embedding for the query
            query_embedding = model.encode([query])[0]
            
            # Calculate similarity
            similarity = cosine_similarity([query_embedding], [job_embedding])[0][0]
            
            # Generate a response based on similarity
            if similarity > 0.5:
                response = f"Based on the job posting, here's what I found:\n\n{structured_info}"
            else:
                response = "I'm sorry, but I couldn't find a relevant answer to your question in the job posting."
            
            st.subheader("Answer")
            st.write(response)
            st.write(f"Confidence: {similarity:.2f}")
    
    except ValueError as e:
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")