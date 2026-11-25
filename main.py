#!/usr/bin/env python
"""
Main entry point for Multi-Document QA System.
Loads PDFs, creates vector store, and enables interactive Q&A.

Usage:
    python main.py                    # Interactive mode (prompts for file paths)
    python main.py file1.pdf file2.pdf  # Use provided PDF files
"""

import os
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

from core.document_loader import load_document
from core.document_splitter import RecursiveCharacterTextSplitter
from core.document_embedding import embedding
from core.document_vector_store import create_vector_store, PERSIST_DIRECTORY, COLLECTION_NAME
from core.answer_generate import rag_chain
from langchain_chroma import Chroma


def setup_vector_store(file_paths):
    """Load, split, and store documents in vector database."""
    print("\n" + "="*60)
    print("SETTING UP VECTOR STORE")
    print("="*60)
    
    # Load documents
    print(f"\nLoading {len(file_paths)} document(s)...")
    documents = load_document(file_paths)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return False
    
    print(f"Total pages loaded: {len(documents)}")
    
    # Split documents
    print("\nSplitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    # Create vector store
    print("\nCreating vector store...")
    try:
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )
        
        # Clear existing collection
        vector_store.delete_collection()
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )
        
        # Add new documents
        vector_store.add_documents(
            documents=chunks,
            ids=[f"document_chunk_{i}" for i in range(len(chunks))]
        )
        
        print(f"Vector store created successfully!")
        print(f"Persisted to: {PERSIST_DIRECTORY}")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total chunks indexed: {len(chunks)}")
        return True
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return False


def interactive_qa():
    """Run interactive Q&A loop."""
    print("\n" + "="*60)
    print("INTERACTIVE Q&A MODE")
    print("="*60)
    print("Type your questions below. Type 'quit' or 'exit' to stop.")
    print("-"*60)
    
    while True:
        try:
            question = input("\n[?] Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print("\n[...] Retrieving relevant documents...")
            print("[...] Generating answer...\n")
            
            answer = rag_chain(question)
            print("-"*60)
            print(answer)
            print("-"*60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def get_file_paths_from_args():
    """Get file paths from command line arguments."""
    file_paths = []
    for arg in sys.argv[1:]:
        path = arg.strip().strip('"')
        if os.path.exists(path) and path.lower().endswith('.pdf'):
            file_paths.append(path)
            print(f"  [OK] Added: {path}")
        else:
            print(f"  [FAIL] Invalid: {path} (file not found or not a PDF)")
    return file_paths


def get_file_paths_interactive():
    """Get file paths interactively from user input."""
    print("\nEnter PDF file paths (one per line).")
    print("Press Enter on empty line when done.")
    print("Example: C:\\Users\\name\\Documents\\file.pdf")
    print("-"*60)
    
    file_paths = []
    while True:
        try:
            path = input("PDF path: ").strip().strip('"')
            if not path:
                break
            if os.path.exists(path) and path.lower().endswith('.pdf'):
                file_paths.append(path)
                print(f"  [OK] Added: {path}")
            else:
                print(f"  [FAIL] Invalid: {path} (file not found or not a PDF)")
        except EOFError:
            break
    return file_paths


def main():
    print("="*60)
    print("MULTI-DOCUMENT QUESTION ANSWERING SYSTEM")
    print("="*60)
    
    # Get file paths from command line args or interactive input
    if len(sys.argv) > 1:
        print("\nUsing file paths from command line arguments:")
        file_paths = get_file_paths_from_args()
    else:
        file_paths = get_file_paths_interactive()
    
    if not file_paths:
        print("\nNo valid PDF files provided. Exiting.")
        return
    
    # Setup vector store
    success = setup_vector_store(file_paths)
    if not success:
        return
    
    # Start interactive Q&A
    interactive_qa()


if __name__ == "__main__":
    main()