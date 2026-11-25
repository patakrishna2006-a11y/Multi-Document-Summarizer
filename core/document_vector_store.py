import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from document_embedding import embedding


PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    "rag_chroma.db"
)

COLLECTION_NAME = "sample"


def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Create and populate vector store with document chunks.
    
    Args:
        chunks: List of LangChain Document objects
        
    Returns:
        Chroma vector store instance
    """
    if not chunks:
        raise ValueError("No document chunks available for indexing.")

    try:
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )

        vector_store.add_documents(
            documents=chunks,
            ids=[f"document_chunk_{i}" for i in range(len(chunks))]
        )

        print(
            f"Vector store generated successfully. "
            f"Total chunks: {len(chunks)}"
        )

        return vector_store

    except Exception as e:
        raise RuntimeError(f"Vector store creation failed: {e}") from e


def get_vector_store() -> Chroma:
    """Get existing vector store (for retrieval)."""
    return Chroma(
        embedding_function=embedding,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )