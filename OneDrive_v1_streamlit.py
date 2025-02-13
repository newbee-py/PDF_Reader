__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import openai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pysqlite3 # type: ignore
import chromadb
from chromadb import PersistentClient
import hashlib
import logging
import concurrent.futures
import dask
from dask import delayed
from dask.distributed import Client

# Set up logging
logging.basicConfig(level=logging.INFO)

# ✅ Disable telemetry to avoid errors
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# ✅ Use persistent ChromaDB client
chroma_client = PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="my_collection")

# ✅ Fix PyTorch threading issues and Force CPU mode
import torch
device = torch.device("cpu")
torch.set_num_threads(1)


# OpenAI API Key
OPENAI_API_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Embedding model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# ChromaDB Client (Persistent Storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="my_collection")

onedrive_folder = r"C:\Users\jatin.malhotra\OneDrive - HCL TECHNOLOGIES LIMITED\Documents\SON Documents\23A.1"
if os.path.exists(onedrive_folder):
    print("Folder exists:", os.listdir(onedrive_folder))  # Lists all files
else:
    print("Folder does not exist!")


# Get PDF files from OneDrive folder
def get_pdf_files(onedrive_folder):
    try:
        pdf_files = [os.path.join(root, file) for root, dirs, files in os.walk(onedrive_folder) if file.endswith(".pdf")]
        logging.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files
    except Exception as e:
        logging.error(f"Error searching OneDrive: {e}")
        return []

# Read PDF contents
def read_pdf_contents(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        logging.info(f"Extracted text from {pdf_file}")
        return text
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_file}: {e}")
        return ""

# Chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    logging.info(f"Chunked text into {len(chunks)} chunks")
    return chunks

# Process individual chunk
def process_chunk(chunk):
    try:
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        embedding = embedding_model.encode(chunk).tolist()

        # Check if chunk already exists
        existing_documents = chroma_collection.get(ids=[chunk_id])
        if existing_documents["ids"]:
            return f"Chunk already exists in the database with ID {chunk_id}"

        chroma_collection.add(ids=[chunk_id], documents=[chunk], embeddings=[embedding])
        return f"Embedded and added chunk to ChromaDB with ID {chunk_id}"
    except Exception as e:
        return f"Error processing chunk: {e}"

# Embed and add to vector database using Dask
def embed_and_add_dask(text_chunks):
    delayed_results = [delayed(process_chunk)(chunk) for chunk in text_chunks]
    results = dask.compute(*delayed_results)
    for result in results:
        logging.info(result)

# Count embeddings in Chroma DB
def count_embeddings():
    try:
        return chroma_collection.count()
    except Exception as e:
        logging.error(f"Error counting embeddings in Chroma DB: {e}")
        return 0

# Query database
def query_database(query, k=5):
    if not query.strip():
        return []
    
    query_embedding = embedding_model.encode(query).tolist()
    try:
        results = chroma_collection.query(query_embeddings=[query_embedding], n_results=k)
        return results["documents"] if "documents" in results else []
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return []

# Generate response with OpenAI
def generate_response(query, context):
    try:
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip() if response.choices else "No response generated."
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error generating response: {e}"

# Streamlit interface
def process_input(onedrive_path, user_input):
    pdf_files = get_pdf_files(onedrive_path)
    if not pdf_files:
        return "No PDFs found in the specified OneDrive folder."

    all_text_chunks = []
    for pdf_file in pdf_files:
        pdf_contents = read_pdf_contents(pdf_file)
        if pdf_contents:
            all_text_chunks.extend(chunk_text(pdf_contents))

    if all_text_chunks:
        embed_and_add_dask(all_text_chunks)
        similar_chunks = query_database(user_input)
        if similar_chunks:
            return generate_response(user_input, "\n".join(similar_chunks))
        return "No similar content found in the database."

    return "No text extracted from PDFs."

st.title("PDF Search and Analysis - OneDrive with OpenAI")

with st.form(key="pdf_search_form"):
    onedrive_path = st.text_input(label="OneDrive Folder Path")
    user_input = st.text_input(label="Query")
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    result = process_input(onedrive_path, user_input)
    st.write(result)
