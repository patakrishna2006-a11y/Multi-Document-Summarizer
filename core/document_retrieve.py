
from langchain_chroma import Chroma
from document_embedding import embedding
import os

# Database configuration
PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    "rag_chroma.db"
)
COLLECTION_NAME = "sample"

# Connect to existing vector database
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME
)

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# User query
query = "What is LangChain?"

# Validate and retrieve documents
if not query.strip():
    raise ValueError("Query cannot be empty.")

try:
    results = retriever.invoke(query)

    if not results:
        print("No relevant documents found.")

    else:
        print("\n========== RETRIEVED DOCUMENTS ==========")

        for i, doc in enumerate(results, start=1):
            print(f"\nDocument {i}")
            print("Content:", doc.page_content)
            print("Metadata:", doc.metadata)

        print("\nTotal Retrieved:", len(results))

except Exception as e:
    print(f"Retrieval failed: {e}")

print("\n========== RETRIEVAL COMPLETED ==========")
