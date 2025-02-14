__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Let the user provide a folder path (default provided)
folder_path = st.text_input("Enter OneDrive Folder Path", value=onedrive_folder)

# Alternatively, let the user upload PDFs directly:
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

# Process files from the folder if provided and exists:
if os.path.exists(folder_path):
    pdf_files = get_pdf_files(folder_path)
    st.write("Found PDF files in folder:", pdf_files)
else:
    pdf_files = []

# Process uploaded PDFs:
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Uploaded files are BytesIO objects; pass them directly to PdfReader
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            st.write(f"Extracted text from {uploaded_file.name}:")
            st.write(text[:500] + "...")
            # Here you can also further process the text (e.g., chunk it, embed, etc.)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")

# You can combine both approaches depending on your application logic.

