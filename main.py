#!/usr/bin/env python
"""
Main entry point for Multi-Document QA System.
Loads PDFs, creates vector store, and enables interactive Q&A.

Usage:
    python main.py                    # Interactive mode (prompts for file paths)
    python main.py file1.pdf file2.pdf  # Use provided PDF files
    python main.py --help             # Show help
    python main.py --rebuild          # Rebuild vector store with same files
"""

import argparse
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


# Default configuration
DEFAULT_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "retrieval_k": 5,
    "persist_directory": PERSIST_DIRECTORY,
    "collection_name": COLLECTION_NAME,
}


def setup_vector_store(file_paths, config=None):
    """Load, split, and store documents in vector database."""
    if config is None:
        config = DEFAULT_CONFIG
    
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
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    # Create vector store
    print("\nCreating vector store...")
    try:
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=config["persist_directory"],
            collection_name=config["collection_name"]
        )
        
        # Clear existing collection
        vector_store.delete_collection()
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=config["persist_directory"],
            collection_name=config["collection_name"]
        )
        
        # Add new documents
        vector_store.add_documents(
            documents=chunks,
            ids=[f"document_chunk_{i}" for i in range(len(chunks))]
        )
        
        print(f"Vector store created successfully!")
        print(f"Persisted to: {config['persist_directory']}")
        print(f"Collection: {config['collection_name']}")
        print(f"Total chunks indexed: {len(chunks)}")
        return True
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return False


def rebuild_vector_store(file_paths, config=None):
    """Rebuild vector store with existing files (same as setup but explicit)."""
    print("\n[...] Rebuilding vector store...")
    return setup_vector_store(file_paths, config)


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


def get_file_paths_from_args(args):
    """Get file paths from command line arguments."""
    file_paths = []
    for arg in args:
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Document QA System with RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive file selection
  python main.py doc1.pdf doc2.pdf                 # Use specific files
  python main.py --rebuild doc1.pdf doc2.pdf       # Rebuild vector store
  python main.py --chunk-size 1000 --overlap 200   # Custom chunk settings
        """
    )
    
    parser.add_argument(
        "files",
        nargs="*",
        help="PDF file paths to process"
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store (clears existing data)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CONFIG["chunk_size"],
        help=f"Chunk size for text splitting (default: {DEFAULT_CONFIG['chunk_size']})"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CONFIG["chunk_overlap"],
        help=f"Chunk overlap for text splitting (default: {DEFAULT_CONFIG['chunk_overlap']})"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_CONFIG["retrieval_k"],
        help=f"Number of chunks to retrieve (default: {DEFAULT_CONFIG['retrieval_k']})"
    )
    
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=DEFAULT_CONFIG["persist_directory"],
        help=f"Vector store persist directory (default: {DEFAULT_CONFIG['persist_directory']})"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_CONFIG["collection_name"],
        help=f"ChromaDB collection name (default: {DEFAULT_CONFIG['collection_name']})"
    )
    
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Only build vector store, skip interactive Q&A"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("MULTI-DOCUMENT QUESTION ANSWERING SYSTEM")
    print("="*60)
    
    # Build config from args
    config = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.overlap,
        "retrieval_k": args.k,
        "persist_directory": args.persist_dir,
        "collection_name": args.collection,
    }
    
    # Get file paths
    if args.files:
        print("\nUsing file paths from command line arguments:")
        file_paths = get_file_paths_from_args(args.files)
    else:
        file_paths = get_file_paths_interactive()
    
    if not file_paths:
        print("\nNo valid PDF files provided. Exiting.")
        return
    
    # Setup or rebuild vector store
    if args.rebuild:
        success = rebuild_vector_store(file_paths, config)
    else:
        success = setup_vector_store(file_paths, config)
    
    if not success:
        return
    
    # Start interactive Q&A (unless --no-qa)
    if not args.no_qa:
        interactive_qa()


if __name__ == "__main__":
    main()