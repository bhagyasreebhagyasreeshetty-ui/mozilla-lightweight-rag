import streamlit as st
import pymupdf4llm
import os
import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer

# Set page configuration for a professional wide layout
st.set_page_config(page_title="Mozilla RAG Dashboard", layout="wide")

UPLOAD_DIR = "uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# --- SIDEBAR DESIGN ---
st.sidebar.title("??? Control Panel")
st.sidebar.markdown("Use this panel to manage your input documents and monitor pipeline metrics.")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document:", type=["pdf"])

if "document_chunks" not in st.session_state:
    st.session_state["document_chunks"] = None
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    
    if st.session_state["faiss_index"] is None:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        with st.sidebar.spinner("Processing Document..."):
            try:
                md_text = pymupdf4llm.to_markdown(file_path)
                chunks = chunk_text(md_text)
                
                embeddings = embedding_model.encode(chunks)
                embeddings_array = np.array(embeddings).astype('float32')
                
                dimension = embeddings_array.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings_array)
                
                st.session_state["document_chunks"] = chunks
                st.session_state["faiss_index"] = index
                st.session_state["md_text_len"] = len(md_text)
                
            except Exception as e:
                st.sidebar.error(f"Error parsing file: {e}")

# --- MAIN SCREEN DESIGN ---
st.title("?? Mozilla Lightweight RAG Application")
st.subheader("Final Year Project Dashboard")
st.markdown("This intelligent assistant parses local documents, creates dense vector representations, and handles semantic retrieval using an entirely offline pipeline.")

if st.session_state["faiss_index"] is not None:
    st.sidebar.success(f"Active File: {uploaded_file.name}")
    
    # Render Metrics cleanly inside the Sidebar
    st.sidebar.markdown("### ?? Pipeline Metrics")
    st.sidebar.metric("Characters Extracted", st.session_state["md_text_len"])
    st.sidebar.metric("Total Vector Chunks", len(st.session_state["document_chunks"]))
    st.sidebar.metric("FAISS Index Database", "Online / Ready")
    
    # Main Chat Interface
    st.markdown("---")
    st.markdown("### ?? Ask Questions to your Document")
    user_query = st.text_input("Type your question below and press Enter:", placeholder="e.g., What are the core skills or projects listed?")
    
    if user_query:
        with st.spinner("Analyzing semantic vectors and generating response..."):
            query_vector = embedding_model.encode([user_query]).astype('float32')
            distances, indices = st.session_state["faiss_index"].search(query_vector, k=1)
            matched_index = indices[0][0]
            retrieved_chunk = st.session_state["document_chunks"][matched_index]
            
            prompt_context = f"""
            You are a helpful AI assistant. Answer the user's question accurately using ONLY the provided document context.
            If the answer cannot be found in the context, politely say you don't know.
            
            Context:
            {retrieved_chunk}
            
            Question: {user_query}
            Answer:
            """
            
            try:
                response = ollama.generate(model="phi3", prompt=prompt_context)
                
                # Display nicely formatted response block
                st.markdown("#### ?? AI Response")
                st.info(response['response'])
                
                with st.expander("?? View Retrieved Source Context"):
                    st.caption(f"Source Document Segment ID: {matched_index}")
                    st.code(retrieved_chunk, language="markdown")
            except Exception as ollama_error:
                st.error(f"Could not connect to Ollama. Verify model 'phi3' is running via terminal! Error: {ollama_error}")
else:
    st.info("?? Please upload a PDF document in the left control panel to activate the RAG AI engine.")
