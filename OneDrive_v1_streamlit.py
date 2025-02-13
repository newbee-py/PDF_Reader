import streamlit as st
import os
import openai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pysqlite3-binary
import chromadb
import hashlib
import logging
import concurrent.futures
import dask
from dask import delayed
from dask.distributed import Client

# Set up logging
logging.basicConfig(level=logging.INFO)

# OpenAI API Key
OPENAI_API_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Embedding model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Vector database
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(name="my_collection")

# Get PDF files from OneDrive folder
def get_pdf_files(onedrive_folder):
    try:
        pdf_files = [os.path.join(root, file) for root, dirs, files in os.walk(onedrive_folder) for file in files if file.endswith(".pdf")]
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
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap")
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if end == len(text) and len(chunk) < chunk_size:
            chunk += " " * (chunk_size - len(chunk))
        chunks.append(chunk)
        start += chunk_size - overlap
    
    logging.info(f"Chunked text into {len(chunks)} chunks")
    return chunks

# Process individual chunk
def process_chunk(chunk):
    try:
        # Create a unique hash for each chunk
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()

        # Embed the chunk
        embedding = embedding_model.encode(chunk).tolist()

        # Check if the chunk already exists in the database
        existing_documents = chroma_collection.query(query_embeddings=[embedding], n_results=1)
        if not existing_documents.get("documents"):
            chroma_collection.add(ids=[chunk_id], documents=[chunk], embeddings=[embedding])
            return f"Embedded and added chunk to vector database with ID {chunk_id}"
        else:
            return f"Chunk already exists in the database with ID {chunk_id}"
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
        all_documents = chroma_collection.get()
        if not all_documents:
            logging.info("No embeddings found in Chroma DB.")
            return 0
        total_embeddings = len(all_documents)
        logging.info(f"Total embeddings in Chroma DB: {total_embeddings}")
        return total_embeddings
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
        return results.get("documents", [])
    except Exception as e:
        logging.error(f"Error querying database: {e}")
        return []

# Generate response with OpenAI
def generate_response(query, context):
    try:
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
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
            return generate_response(user_input, similar_chunks[0])
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
