FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces runs on port 7860 by default
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
