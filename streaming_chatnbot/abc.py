# app.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import pdfplumber
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

# Load the SentenceTransformer model
model = SentenceTransformer("meta-llama/Meta-Llama-3-8B")

# Initialize FAISS vector store
vector_store = FAISS()

# Function to process PDF files
def process_pdf(file_path):
    texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return texts

# Function to process CSV files
def process_csv(file_path):
    df = pd.read_csv(file_path)
    texts = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
    return texts

# Function to embed texts
def embed_texts(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings.numpy()

# Route to upload files
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    if file.filename.endswith('.pdf'):
        texts = process_pdf(file_path)
    elif file.filename.endswith('.csv'):
        texts = process_csv(file_path)
    else:
        return jsonify({"error": "Unsupported file type"})
    
    embeddings = embed_texts(texts)
    vector_store.add(embeddings, texts)
    
    return jsonify({"message": "File processed and added to vector store"})

# Route to handle queries
@app.route('/query', methods=['POST'])
def query():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"})
    
    query_embedding = embed_texts([query])[0]
    results = vector_store.query(query_embedding, top_k=5)
    
    return jsonify({"results": results})

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
