import streamlit as st
import networkx as nx
from pyvis.network import Network

def plot_network_graph(df, keyword):
    """
    Plot a network graph of accounts sharing a specific keyword.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing Reddit posts
    keyword (str): Keyword to filter posts
    """
    # Filter posts containing the keyword
    df_filtered = df[df['title'].fillna('').str.contains(keyword, case=False, na=False)]
    
    if df_filtered.empty:
        st.warning(f"No posts found containing the keyword '{keyword}'")
        return
    
    # Create a network graph
    G = nx.Graph()
    for _, row in df_filtered.iterrows():
        G.add_node(row['author'], label=row['author'])
        G.add_edge(row['author'], row['subreddit'])
    
    # Visualize the graph using PyVis
    net = Network(notebook=True, height="500px", width="100%")
    net.from_nx(G)
    net.show("network.html")
    
    try:
        with open("network.html", "r") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=500)
    except FileNotFoundError:
        st.error("Network visualization file not found. This might be due to environment restrictions.")