import requests
import streamlit as st
def fetch_job_posting_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        st.write(response.text)
        return response.text
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch the job posting: {str(e)}")