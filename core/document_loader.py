import os 
from langchain_community.document_loaders import PyPDFLoader

def load_document(file_paths):

    documents = []

    for file_path in file_paths:

        try:
             # Check whether the file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # Check PDF extension
            if not file_path.lower().endswith(".pdf"):
                print(f"Invalid file format: {file_path}")
                continue
            
            loader = PyPDFLoader(file_path)

            pages = loader.load()

            documents.extend(pages)

            print(f"Document: {file_path}")
            print(f"Pages: {len(pages)}")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents