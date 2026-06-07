FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl procps && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set up a new user named "user" with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_MODELS=/home/user/.ollama/models

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Start Ollama in background and pull models during build
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2 && \
    ollama pull nomic-embed-text

# Copy the rest of the application
COPY --chown=user . .

# Expose port (HF Spaces defaults to 7860)
ENV PORT=7860
EXPOSE 7860

# Start Ollama in background, wait, then launch app
CMD ["sh", "-c", "ollama serve & sleep 5 && python app.py"]
