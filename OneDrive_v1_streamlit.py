__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import hashlib
import logging
import chromadb
import dask
import openai
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  #Disable GPU
import tensorflow_hub as hub
# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
from chromadb import PersistentClient
from dask import delayed
from PyPDF2 import PdfReader
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)

# Disable telemetry to avoid errors
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# OpenAI API Key
OPENAI_API_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Sample text for embedding
sample_text = "This is a sample text for testing."
embedding = embed([sample_text]).numpy().tolist()[0]
st.write("Sample Embedding:", embedding)
         
# Use persistent ChromaDB client without tenant
chroma_client = PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="my_collection")

# Set of known chunk IDs for deduplication
known_chunk_ids = set()

# Get PDF files from OneDrive folder
def get_pdf_files(folder_path):
    """
    Get a list of PDF files from the specified OneDrive folder.
    """
    try:
        pdf_files = []
        for root, dirs, files in os.walk(folder_path):
            print(f"Scanning directory: {root}")
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, filename))
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
        
        # Encode the chunk using Universal Sentence Encoder
        embedding = embed([chunk]).numpy().tolist()[0]

        # Check if chunk already exists
        if chunk_id not in known_chunk_ids:
            chroma_collection.add(ids=[chunk_id], documents=[chunk], embeddings=[embedding])
            known_chunk_ids.add(chunk_id)  # Add to known IDs
            return f"Embedded and added chunk to ChromaDB with ID {chunk_id}"
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
        return chroma_collection.count()
    except Exception as e:
        logging.error(f"Error counting embeddings in Chroma DB: {e}")
        return 0

# Query database
def query_database(query, k=5):
    if not query.strip():
        return []

    query_embedding = embed([query]).numpy().tolist()[0]

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
            return generate_response(user_input, "\n".join(similar_chunks))
        return "No similar content found in the database."

    return "No text extracted from PDFs."

# Define OneDrive folder path using os.path.join
# for cross-platform compatibility, but let the user override it
default_onedrive_folder = os.path.join(
    "C:", "Users", "jatin.malhotra", "OneDrive - HCL TECHNOLOGIES LIMITED", "Documents", "SON Documents", "23A.1"
)

st.title("PDF Search and Analysis - OneDrive with OpenAI")

with st.form(key="pdf_search_form"):
    onedrive_path_input = st.text_input(label="OneDrive Folder Path", value=default_onedrive_folder)
    user_input = st.text_input(label="Query")
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    result = process_input(onedrive_path_input, user_input)
    st.write(result)
