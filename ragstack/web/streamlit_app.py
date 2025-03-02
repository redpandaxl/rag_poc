"""
Streamlit web UI for the RAG application.
"""
import os
import requests
import streamlit as st

from ragstack.config.settings import settings


# Set Streamlit page config
st.set_page_config(
    page_title="RAGStack",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API URL
API_URL = f"http://{settings.api_host}:{settings.api_port}"


def search_documents(query, top_k=5):
    """Search for documents via the API."""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()["results"]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []


def upload_file(file):
    """Upload a file via the API."""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def get_collection_stats(collection=None):
    """Get collection statistics via the API."""
    try:
        params = {}
        if collection:
            params["collection"] = collection
        
        response = requests.get(f"{API_URL}/stats", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to get stats: {str(e)}")
        return None


def list_collections():
    """List all collections via the API."""
    try:
        response = requests.get(f"{API_URL}/collections")
        response.raise_for_status()
        return response.json().get("collections", [])
    except Exception as e:
        st.error(f"Failed to list collections: {str(e)}")
        return []


def main():
    """Main Streamlit application."""
    st.title("RAGStack: Retrieval-Augmented Generation")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Search", "Upload", "Stats"])
    
    if page == "Search":
        show_search_page()
    elif page == "Upload":
        show_upload_page()
    elif page == "Stats":
        show_stats_page()


def show_search_page():
    """Show the search page."""
    st.header("Document Search")
    
    query = st.text_input("Enter your query:")
    top_k = st.slider("Number of results", 1, 20, 5)
    
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            results = search_documents(query, top_k)
        
        if results:
            st.success(f"Found {len(results)} results")
            
            for i, result in enumerate(results):
                with st.expander(f"Result {i+1}: {result['metadata'].get('title', 'Untitled')}"):
                    # Format metadata
                    metadata = result["metadata"]
                    source = metadata.get("source", "Unknown")
                    tags = ", ".join(metadata.get("tags", []))
                    summary = metadata.get("summary", "")
                    
                    # Display metadata
                    st.markdown(f"**Source:** {source}")
                    if tags:
                        st.markdown(f"**Tags:** {tags}")
                    if summary:
                        st.markdown(f"**Summary:** {summary}")
                    
                    # Display content
                    st.markdown("### Content")
                    st.markdown(result["content"])
                    
                    # Display similarity score
                    st.markdown(f"**Similarity Score:** {result['score']:.4f}")
        else:
            st.warning("No results found. Try a different query.")


def show_upload_page():
    """Show the upload page."""
    st.header("Upload Documents")
    
    uploaded_file = st.file_uploader("Choose a file to upload", 
                                    type=["txt", "pdf", "docx", "doc", "xlsx", "xls", "csv", "md", "json"])
    
    if uploaded_file is not None:
        if st.button("Upload and Process"):
            with st.spinner("Uploading..."):
                result = upload_file(uploaded_file)
            
            if result and result["status"] == "success":
                st.success(f"File '{result['filename']}' uploaded successfully!")
                st.info("The file has been queued for processing. It will be available for search once processing is complete.")
            else:
                st.error("Upload failed.")


def show_stats_page():
    """Show the statistics page."""
    st.header("Vector Database Statistics")
    
    # Get list of collections
    collections = list_collections()
    
    if collections:
        selected_collection = st.selectbox("Select collection", collections)
        
        if st.button("Get Statistics"):
            with st.spinner("Loading statistics..."):
                stats = get_collection_stats(selected_collection)
            
            if stats:
                st.subheader(f"Statistics for {stats['collection']}")
                
                # Display stats as a table
                stats_data = stats["stats"]
                st.json(stats_data)
    else:
        st.warning("No collections found.")


def start_streamlit():
    """Start the Streamlit app."""
    import subprocess
    import sys
    import os
    
    # Run streamlit directly to avoid any issues with subprocess
    os.environ["STREAMLIT_SERVER_PORT"] = str(settings.web_port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = settings.web_host
    
    # Import streamlit functions directly
    import streamlit.web.bootstrap as bootstrap
    
    # Get the script path
    script_path = os.path.abspath(__file__)
    
    # Run the Streamlit app directly
    bootstrap.run(script_path, "", [], flag_options={})


if __name__ == "__main__":
    main()