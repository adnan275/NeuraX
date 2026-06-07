FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl procps zstd && rm -rf /var/lib/apt/lists/*

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
RUN mkdir -p /home/user/.ollama/models && \
    ollama serve & \
    for i in $(seq 1 60); do \
        if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then \
            echo "Ollama server started successfully."; \
            break; \
        fi; \
        echo "Waiting for Ollama server to start (attempt $i)..."; \
        sleep 1; \
    done && \
    ollama pull llama3.2 && \
    ollama pull nomic-embed-text && \
    pkill ollama

# Copy the rest of the application
COPY --chown=user . .

# Expose port (HF Spaces defaults to 7860)
ENV PORT=7860
EXPOSE 7860

# Start Ollama in background, wait for it, then launch app
CMD ["sh", "-c", "ollama serve & for i in $(seq 1 60); do if curl -s http://127.0.0.1:11434/api/tags >/dev/null; then break; fi; sleep 1; done && python app.py"]
