import streamlit as st
import os
import pymupdf4llm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import httpx

st.set_page_config(page_title="Mozilla Lightweight RAG", page_icon="🤖", layout="wide")
st.title(" Mozilla Lightweight RAG System")
st.write("A completely local, privacy-preserving document intelligence application.")

if "messages" not in st.session_state: st.session_state.messages = []
if "vector_index" not in st.session_state: st.session_state.vector_index = None
if "chunks" not in st.session_state: st.session_state.chunks = []

@st.cache_resource
def load_embedding_model(): 
    return SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = load_embedding_model()

st.sidebar.header("📁 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload your Reference PDF", type=["pdf"])

if uploaded_file and st.session_state.vector_index is None:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f: 
        f.write(uploaded_file.getbuffer())
    with st.spinner("Parsing and embedding document structures..."):
        try:
            md_text = pymupdf4llm.to_markdown(temp_path)
            chunk_size, overlap = 1000, 200
            chunks = [md_text[i:i+chunk_size] for i in range(0, len(md_text), chunk_size - overlap)]
            st.session_state.chunks = chunks
            embeddings = embedding_model.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype("float32"))
            st.session_state.vector_index = index
            st.sidebar.success(f"Successfully processed {len(chunks)} chunks!")
        except Exception as e: 
            st.sidebar.error(f"Error: {e}")
        finally:
            if os.path.exists(temp_path): 
                os.remove(temp_path)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about your uploaded document:"):
    with st.chat_message("user"): 
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    if st.session_state.vector_index is None:
        with st.chat_message("assistant"): 
            st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching and synthesizing answer..."):
                query_vector = embedding_model.encode([user_query])
                D, I = st.session_state.vector_index.search(np.array(query_vector).astype("float32"), k=3)
                retrieved_context = "\n".join([st.session_state.chunks[idx] for idx in I[0] if idx < len(st.session_state.chunks)])
                try:
                    # CHANGED: timeout=None allows slow laptops to load the model without crashing
                    res = httpx.post(
                        "http://localhost:11434/api/generate", 
                        json={"model": "llama3.2:1b", "prompt": f"Context:\n{retrieved_context}\n\nQuery: {user_query}", "stream": False}, 
                        timeout=None
                    )
                    ai_res = res.json().get("response", "Error.")
                    st.markdown(ai_res)
                    st.session_state.messages.append({"role": "assistant", "content": ai_res})
                except Exception as e: 
                    st.error(f"Ollama error: {e}. Trying to download ultra-lightweight model...")