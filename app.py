import streamlit as st
import os
import pymupdf4llm
import ollama

st.set_page_config(page_title="Mozilla Blueprint RAG", layout="centered")
st.title("📚 Structural Markdown-Based Chatbot")
st.write("A lightweight, vector-free implementation based on the Mozilla.ai RAG Blueprint.")

# Target Directory
UPLOAD_DIR = "uploaded_docs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved {uploaded_file.name} successfully!")

    # 1. Structural Extraction (No Vector Database / Embeddings)
    with st.spinner("Extracting layout boundaries via PyMuPDF4LLM..."):
        md_text = pymupdf4llm.to_markdown(file_path)
        
        # Parse Markdown headers into discrete semantic components
        sections = {}
        current_heading = "Overview"
        current_content = []
        
        for line in md_text.split("\n"):
            if line.strip().startswith(("# ", "## ", "### ")):
                if current_content:
                    sections[current_heading] = "\n".join(current_content).strip()
                current_heading = line.replace("#", "").strip()
                current_content = [line]
            else:
                current_content.append(line)
        if current_content:
            sections[current_heading] = "\n".join(current_content).strip()
            
    st.info(f"Document segmented into {len(sections)} layout sections!")

    # Chat UI
    user_query = st.text_input("Ask a question about the document:")
    
    if user_query:
        with st.spinner("Routing query to the correct document layout section..."):
            # 2. Roaming RAG Router Pattern
            titles_list = "\n".join([f"- {title}" for title in sections.keys()])
            router_prompt = (
                f"Select the single most relevant section title from this list that contains "
                f"the answer to the user's question.\n\nList:\n{titles_list}\n\nQuestion: {user_query}\n\n"
                f"Respond ONLY with the exact chosen title name."
            )
            
            route_res = ollama.chat(
                model="qwen2.5:0.5b",
                messages=[{"role": "user", "content": router_prompt}],
                options={"temperature": 0.0}
            )
            chosen_title = route_res['message']['content'].strip()
            
            # Fallback evaluation matcher
            if chosen_title not in sections:
                matched = [t for t in sections.keys() if t.lower() in chosen_title.lower()]
                chosen_title = matched[0] if matched else list(sections.keys())[0]
                
            st.caption(f"📍 Content pulled exclusively from section: **{chosen_title}**")
            
        with st.spinner("Formulating localized answer..."):
            # 3. Targeted Context Processing
            context_block = sections[chosen_title]
            qa_prompt = (
                f"Answer the question strictly using this context block. If it doesn't contain the answer, "
                f"say 'Information not found.'\n\nContext:\n{context_block}\n\nQuestion: {user_query}"
            )
            
            ans_res = ollama.chat(
                model="qwen2.5:0.5b",
                messages=[{"role": "user", "content": qa_prompt}],
                options={"temperature": 0.2}
            )
            
            st.write("### Answer:")
            st.write(ans_res['message']['content'].strip())