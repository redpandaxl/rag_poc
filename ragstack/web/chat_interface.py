"""
ChatGPT-like interface for the RAG application using Streamlit.
This interface connects to the vector database and provides a conversational UI.
"""
import os
import requests
import streamlit as st
import json
from typing import List, Dict, Any

from ragstack.config.settings import settings
from ragstack.core.rag import RAGEngine

# Set Streamlit page config
st.set_page_config(
    page_title="RAG Chat Interface",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize RAG engine
@st.cache_resource
def get_rag_engine():
    """Get or create the RAG engine instance."""
    return RAGEngine()

# Message history management
def get_chat_history():
    """Get chat history from session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages

def add_message(role: str, content: str, metadata: Dict[str, Any] = None):
    """Add a message to the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    message = {"role": role, "content": content}
    if metadata:
        message["metadata"] = metadata
    
    st.session_state.messages.append(message)

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.messages = []

# These functions are no longer needed since we're using the enhanced RAG engine
# functionality directly. They're kept here for reference but not used anymore.

def _format_prompt_with_context(query: str, context: str) -> str:
    """Format a prompt for the LLM with the retrieved context."""
    prompt = f"""You are an intelligent assistant helping with document queries. 
Use ONLY the provided CONTEXT to answer the QUESTION. If the answer cannot be 
determined from the CONTEXT, respond with "I don't have enough information to 
answer that question based on the available documents."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

def _send_to_llm(prompt: str) -> str:
    """Send a prompt to the LLM and get a response."""
    # Try to use the configured LLM from settings
    try:
        from ragstack.config.settings import settings
        import requests
        
        llm_settings = settings.llm_settings
        
        # Handle different LLM backends
        if "host" in llm_settings and "model_name" in llm_settings:
            # Assuming Ollama-like API
            ollama_url = f"{llm_settings['host']}/api/generate"
            
            response = requests.post(
                ollama_url,
                json={
                    "model": llm_settings["model_name"],
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Error: No response from LLM")
            else:
                return f"Error connecting to LLM: {response.status_code}"
                
    except Exception as e:
        return f"Error processing with LLM: {str(e)}"
    
    # Fallback to a simple response
    return "Unable to connect to an LLM for response generation."

def process_query(query: str, use_llm: bool = True):
    """Process a user query with RAG."""
    rag_engine = get_rag_engine()
    
    # Get chat history from session state
    chat_history = []
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            if msg["role"] in ["user", "assistant"]:
                chat_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
    
    if use_llm:
        # Use the enhanced conversational processing
        result = rag_engine.process_conversational_query(
            query=query,
            chat_history=chat_history,
            temperature=0.7
        )
        
        # Extract components from result
        response = result["response"]
        sources = result["sources"]
        context = result["context"]
        
        # Format sources for display
        source_list = []
        for source in sources:
            source_list.append(f"Document {source['index']}: {source['title']} ({source['source']})")
        
        return {
            "response": response,
            "sources": source_list,
            "context": context
        }
    else:
        # Just return the retrieved context without LLM processing
        context = rag_engine.get_retrieval_context(query)
        
        # Parse sources from context
        sources = []
        if context:
            # Find all document references in the context
            import re
            doc_matches = re.findall(r'\[Document (\d+)\]', context)
            sources = [f"Document {match}" for match in doc_matches]
        
        if context:
            return {
                "response": f"Here's what I found based on your query:\n\n{context}",
                "sources": sources,
                "context": context
            }
        else:
            return {
                "response": "I couldn't find any relevant information for your query.",
                "sources": [],
                "context": ""
            }
    }

def main():
    """Main Streamlit application."""
    st.title("RAG Chat Interface")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Show available collections
        rag_engine = get_rag_engine()
        try:
            collections = rag_engine.vector_db.list_collections()
            if collections:
                selected_collection = st.selectbox(
                    "Select Collection", 
                    options=collections,
                    index=collections.index(rag_engine.collection_name) if rag_engine.collection_name in collections else 0
                )
                
                # Update collection if changed
                if selected_collection != rag_engine.collection_name:
                    rag_engine.collection_name = selected_collection
                    st.success(f"Switched to collection: {selected_collection}")
        except Exception as e:
            st.error(f"Error loading collections: {str(e)}")
        
        # Set number of results to retrieve
        top_k = st.slider("Number of results to retrieve", min_value=1, max_value=20, value=5)
        if rag_engine.top_k != top_k:
            rag_engine.top_k = top_k
            
        # LLM Settings
        st.subheader("LLM Settings")
        
        # Option to use LLM for response generation
        if "use_llm" not in st.session_state:
            st.session_state.use_llm = True
            
        use_llm = st.toggle("Use LLM for responses", value=st.session_state.use_llm)
        if use_llm != st.session_state.use_llm:
            st.session_state.use_llm = use_llm
        
        # Temperature slider (only shown when LLM is enabled)
        if use_llm:
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.1,
                help="Higher values make output more creative, lower values more deterministic"
            )
            if "temperature" not in st.session_state or st.session_state.temperature != temperature:
                st.session_state.temperature = temperature
            
        # Show raw context toggle
        if "show_context" not in st.session_state:
            st.session_state.show_context = False
            
        show_context = st.toggle("Show raw context", value=st.session_state.show_context)
        if show_context != st.session_state.show_context:
            st.session_state.show_context = show_context
        
        # Show memory toggle for conversation history
        if "use_memory" not in st.session_state:
            st.session_state.use_memory = True
            
        use_memory = st.toggle(
            "Use conversation memory", 
            value=st.session_state.use_memory,
            help="When enabled, the assistant will remember previous exchanges in the conversation"
        )
        if use_memory != st.session_state.use_memory:
            st.session_state.use_memory = use_memory
        
        # Add clear chat button
        if st.button("Clear Chat"):
            clear_chat_history()
            st.rerun()
    
    # Display chat messages
    messages = get_chat_history()
    for message in messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            
            # Only show these for assistant messages
            if message["role"] == "assistant" and message.get("metadata"):
                # Display raw context if available and enabled
                if st.session_state.show_context and message["metadata"].get("context"):
                    with st.expander("Raw Context"):
                        st.markdown(message["metadata"]["context"])
                
                # Display sources if available
                if message["metadata"].get("sources"):
                    with st.expander("Sources"):
                        for source in message["metadata"]["sources"]:
                            st.markdown(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                # Get chat history if memory is enabled
                chat_history = get_chat_history() if st.session_state.use_memory else []
                
                # Get temperature if set
                temperature = st.session_state.get("temperature", 0.7) if st.session_state.use_llm else None
                
                # Process the query
                result = process_query(prompt, use_llm=st.session_state.use_llm)
                response = result["response"]
                sources = result.get("sources", [])
                context = result.get("context", "")
            
            st.markdown(response)
            
            # Display raw context if enabled
            if st.session_state.show_context and context:
                with st.expander("Raw Context"):
                    st.markdown(context)
            
            # Display sources if available
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")
        
        # Add assistant message to chat
        add_message("assistant", response, {
            "sources": sources,
            "context": context if st.session_state.show_context else None
        })

def start_chat_interface():
    """Start the chat interface."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = str(settings.web_port + 1)  # Use a different port than main app
    os.environ["STREAMLIT_SERVER_ADDRESS"] = settings.web_host
    
    # Get the script path
    script_path = os.path.abspath(__file__)
    
    # Use subprocess to run streamlit directly (more reliable than bootstrap)
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", script_path]
    
    print(f"Starting chat interface with command: {' '.join(streamlit_cmd)}")
    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        print("Chat interface stopped by user")
    except Exception as e:
        print(f"Error running chat interface: {str(e)}")

if __name__ == "__main__":
    main()