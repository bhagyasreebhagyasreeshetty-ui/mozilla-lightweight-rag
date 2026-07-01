# Use an official lightweight Python image
FROM python:3.10-slim

# Install system dependencies required for FAISS and PDF parsing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama inside the container to handle the local LLM
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory inside container
WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Start Ollama in the background, download phi3, and run Streamlit
CMD ollama serve & sleep 5 && ollama pull phi3 && streamlit run app.py --server.port=8501 --server.address=0.0.0.0
