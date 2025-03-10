import streamlit as st
import pandas as pd
import json
import os

def load_dataset():
    """
    Load the dataset from a local JSONL file in the data folder.
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    try:
        # Define the path to the data file
        file_path = os.path.join('data', 'data.jsonl')
        
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"Data file not found at {file_path}. Please make sure 'data.jsonl' exists in the 'data' folder.")
            return None
            
        # Load the JSONL file
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
            
        # Create DataFrame
        df = pd.DataFrame([post['data'] for post in data])
        
        st.success(f"Successfully loaded {len(df)} posts from data.jsonl")
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None