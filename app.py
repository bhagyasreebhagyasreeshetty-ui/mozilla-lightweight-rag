import streamlit as st
import os
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
import faiss

# 1. Configure Streamlit Page Settings
st.set_page_config(page_title="Mozilla Lightweight RAG App", layout="centered")
st.title("🦙 Deployable Mozilla Lightweight RAG")
st.write("A production-ready RAG application. Built with LlamaIndex & Hugging Face.")

# 2. Secure API Key Management 
# In production, add your HF_TOKEN token to Streamlit's secrets manager!
hf_token = st.sidebar.text_input("Enter Hugging Face Token", type="password", value=os.getenv("HF_TOKEN", ""))

if not hf_token:
    st.info("💡 Please enter your Hugging Face API Token in the sidebar to begin. (It is completely free to create on huggingface.co)")
    st.stop()

# Initialize lightweight, cloud-hosted AI models via Hugging Face API
@st.cache_resource
def init_cloud_models(token):
    # Free, incredibly capable models hosted via HF's serverless pipeline
    llm = HuggingFaceInferenceAPI(model_name="meta-llama/Meta-Llama-3-8B-Instruct", token=token)
    embed_model = HuggingFaceInferenceAPIEmbedding(model_name="BAAI/bge-small-en-v1.5", token=token)
    return llm, embed_model

llm, embed_model = init_cloud_models(hf_token)

# 3. File Upload Section
uploaded_file = st.file_uploader("Upload your target PDF document", type=["pdf"])

if uploaded_file is not None:
    temp_dir = "temp_docs"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"Saved {uploaded_file.name} successfully!")

    # 4. Processing Pipeline
    with st.spinner("Parsing PDF layout and building vector index via cloud embedding engine..."):
        try:
            # Step A: Parse PDF layout
            md_text = pymupdf4llm.to_markdown(file_path)
            
            # Step B: Chunk text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(md_text)
            
            # Step C: Convert to Documents
            documents = [Document(text=chunk) for chunk in chunks]
            
            # Step D: Construct FAISS Index (bge-small-en-v1.5 uses 384 dimensions)
            d = 384  
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Step E: Build index
            index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context, 
                embed_model=embed_model
            )
            st.success("Indexing completed! Your cloud vector space is active.")
            
            # 5. Query Interface
            st.write("---")
            st.subheader("Ask questions about your document")
            query_engine = index.as_query_engine(llm=llm)
            
            user_query = st.text_input("Enter your question here:")
            if user_query:
                with st.spinner("Streaming response from LLM..."):
                    response = query_engine.query(user_query)
                    st.markdown(f"**Answer:** {response}")
                    
        except Exception as e:
            st.error(f"Pipeline error: {e}")