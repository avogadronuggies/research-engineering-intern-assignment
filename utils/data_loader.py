import streamlit as st
import pandas as pd
import json

def load_dataset(file):
    """
    Load the dataset from a file (JSONL or CSV).
    
    Parameters:
    file: Uploaded file object
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    if file.name.endswith('.jsonl'):
        data = [json.loads(line) for line in file]
        df = pd.DataFrame([post['data'] for post in data])
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        st.error("Unsupported file format. Please upload a JSONL or CSV file.")
        return None
    return df