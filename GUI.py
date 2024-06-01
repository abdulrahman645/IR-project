import streamlit as st
import requests
import json
import time
import streamlit.components.v1 as components

# Define the datasets
datasets = ["sara", "antiquetrain", "wikiren1ktraining"]

# Create a selectbox for the datasets
dataset = st.selectbox('Select a dataset:', datasets)

# Create a text input for the search query
query = st.text_input('Enter your search query:', key="query_input")


# Custom component for detecting changes in the query input
components.html("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    var queryInput = document.querySelector('#query_input');
    queryInput.addEventListener('input', function() {
        // Notify Streamlit of the change
        streamlit.setComponentReady();
    });
});
</script>
""", height=0)

# Function to fetch suggestions
def fetch_suggestions():
    response = requests.get(f"http://127.0.0.1:8000/search/realtime_suggestions/{dataset}", params={"query": query})
    return response.json()["suggestions"]

# Use Streamlit's session state to cache the suggestions
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Update suggestions based on the current query
if query!= "":
    st.session_state.suggestions = fetch_suggestions()

# Display the suggestions
for i, suggestion in enumerate(st.session_state.suggestions, start=1):
    st.write(f"Suggestion {i}: {suggestion}")
    st.markdown("---")

# Button to trigger the main search
if st.button('Search'):
    # Send a GET request to the search API
    response = requests.get(f"http://127.0.0.1:8000/search/{dataset}", params={"query": query})

    # Parse the JSON response
    results = response.json()

    # Display the retrieved documents
    for i, doc in enumerate(results["retrieved"], start=1):
        # Check if the document is too long
        if len(doc) > 300:
            # Display a short preview with a "Read More" button
            preview = doc[:300]
            with st.expander(f"Document {i}: {preview}... (click to expand)"):
                st.write(doc)
        else:
            # If the document is short, display it as is
            st.write(f"Document {i}: {doc}")
        
        # Add a horizontal line for separation
        st.markdown("---")

    # Display the "Read Also" section
    st.header("Read Also")
    for i, doc in enumerate(results["cluster_doc"], start=1):
        # Check if the document is too long
        if len(doc) > 300:
            # Display a short preview with a "Read More" button
            preview = doc[:300]
            with st.expander(f"Suggestion {i}: {preview}... (click to expand)"):
                st.write(doc)
        else:
            # If the document is short, display it as is
            st.write(f"Suggestion {i}: {doc}")

        # Add a horizontal line for separation
        st.markdown("---")
