from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Re-export for convenience
__all__ = ["RecursiveCharacterTextSplitter", "split_documents"]


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Split documents into chunks.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Size of each chunk (default: 800)
        chunk_overlap: Overlap between chunks (default: 100)
        
    Returns:
        List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)