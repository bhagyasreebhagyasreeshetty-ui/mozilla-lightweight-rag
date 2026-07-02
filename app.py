import subprocess
import sys

# 1. Force install missing dependencies on the Streamlit Cloud server
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-text-splitters", "langchain"])
    from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
import os
import pymupdf4llm
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import faiss

# 2. Configure Streamlit Page Settings
st.set_page_config(page_title="Mozilla Lightweight RAG App", layout="centered")
st.title("🦙 Mozilla Lightweight RAG Application")
st.write("Upload a PDF document to parse it and ask questions using your local/cloud LLM environment.")

# 3. Sidebar Configuration for Models
st.sidebar.header("Configuration")
llm_model = st.sidebar.selectbox("Select LLM Model", ["llama3", "mistral", "phi3"], index=0)
embed_model_name = st.sidebar.selectbox("Select Embedding Model", ["nomic-embed-text", "bge-small-en"], index=0)

# Initialize Ollama LLM & Embedding models
@st.cache_resource
def init_models(llm_name, embed_name):
    llm = Ollama(model=llm_name, request_timeout=60.0)
    embed_model = OllamaEmbedding(model_name=embed_name)
    return llm, embed_model

try:
    llm, embed_model = init_models(llm_model, embed_model_name)
except Exception as e:
    st.sidebar.warning("Could not connect to a local Ollama instance. Ensure Ollama is running.")

# 4. File Upload Section
uploaded_file = st.file_file_uploader("Upload your target PDF document", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_dir = "temp_docs"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"Saved {uploaded_file.name} successfully!")

    # 5. Process & Indexing Pipeline
    with st.spinner("Parsing PDF and building vector index..."):
        try:
            # Step A: Parse PDF to Markdown layout text using PyMuPDF4LLM
            md_text = pymupdf4llm.to_markdown(file_path)
            
            # Step B: Chunk the text beautifully using the recursive character splitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(md_text)
            
            # Step C: Convert chunks to LlamaIndex Documents
            documents = [Document(text=chunk) for chunk in chunks]
            
            # Step D: Construct a FAISS Vector Index
            d = 4096 if "nomic" in embed_model_name else 384  # Adjust dimensions based on model choice
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Step E: Build the index
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                embed_model=embed_model
            )
            st.success("Indexing completed! Your document is ready for queries.")
            
            # 6. Query Interface
            st.write("---")
            st.subheader("Ask questions about your document")
            query_engine = index.as_query_engine(llm=llm)
            
            user_query = st.text_input("Enter your question here:")
            if user_query:
                with st.spinner("Thinking..."):
                    response = query_engine.query(user_query)
                    st.markdown(f"**Answer:** {response}")
                    
        except Exception as e:
            st.error(f"An error occurred during pipeline execution: {e}")