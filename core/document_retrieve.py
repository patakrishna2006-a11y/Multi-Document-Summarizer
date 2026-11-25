from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from document_vector_store import get_vector_store


def get_retriever() -> BaseRetriever:
    """Get a fresh retriever from the current vector store."""
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )


def retrieve_documents(query: str) -> List[Document]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        query: Search query string
        
    Returns:
        List of retrieved Document objects
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    try:
        retriever = get_retriever()
        results = retriever.invoke(query)

        if not results:
            print("No relevant documents found.")
            return []

        print("\n========== RETRIEVED DOCUMENTS ==========")

        for i, doc in enumerate(results, start=1):
            print(f"\nDocument {i}")
            print("Content:", doc.page_content)
            print("Metadata:", doc.metadata)

        print("\nTotal Retrieved:", len(results))

        return results

    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {e}") from e