import streamlit as st
import httpx
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set up the Streamlit Page Layout
st.set_page_config(page_title="Local Resume RAG AI", layout="centered")
st.title("📄 Local Resume RAG Assistant")
st.write("Ask questions about your uploaded resume completely offline.")

# 1. Initialize the Embedding Model (Cached so it stays fast)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Document Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# 2. Process the PDF and Create Vector Store if Uploaded
vector_store = None
if uploaded_file is not None:
    # Save uploaded file temporarily to read it
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Parsing PDF and indexing sections..."):
        try:
            # Extract clean markdown text from PDF
            md_text = pymupdf4llm.to_markdown("temp_resume.pdf")
            
            # Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(md_text)
            
            # Build the FAISS Vector Database locally
            vector_store = FAISS.from_texts(chunks, embeddings)
            st.sidebar.success("Resume processed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error parsing PDF: {e}")

# 3. Display Existing Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle User Input
if user_query := st.chat_input("Ask something about the resume..."):
    # Display the user's question instantly
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Response using RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if vector_store is None:
                ai_res = "Please upload a resume in the sidebar first before asking questions!"
                st.markdown(ai_res)
                st.session_state.messages.append({"role": "assistant", "content": ai_res})
            else:
                try:
                    # Retrieve the top 3 most relevant chunks from FAISS
                    docs = vector_store.similarity_search(user_query, k=3)
                    retrieved_context = "\n---\n".join([doc.page_content for doc in docs])
                    
                    # Create the payload exactly with streaming disabled
                    payload = {
                        "model": "llama3.2:1b",
                        "prompt": f"Context:\n{retrieved_context}\n\nQuery: {user_query}\n\nAnswer the query clearly based ONLY on the context provided above.",
                        "stream": False  # <--- Fixes the infinite loading bug!
                    }
                    
                    # Send request to local Ollama API endpoint
                    res = httpx.post("http://localhost:11434/api/generate", json=payload, timeout=None)
                    
                    # Extract the fully built message response from Ollama
                    ai_res = res.json()["response"]
                    
                    st.markdown(ai_res)
                    st.session_state.messages.append({"role": "assistant", "content": ai_res})
                    
                except Exception as e:
                    error_msg = f"Error connecting to Ollama: {e}. Please ensure Ollama is running."
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})