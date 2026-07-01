# Use an official lightweight Python runtime
FROM python:3.10-slim

# Install system dependencies required for FAISS and network tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama internally to handle the local LLM execution layer
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory inside container
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose Streamlit's default dashboard network port
EXPOSE 8501

# Boot Ollama backend engine, pull the phi3 model, and spin up your Streamlit UI
CMD ollama serve & sleep 5 && ollama pull phi3 && streamlit run app.py --server.port=8501 --server.address=0.0.0.0
