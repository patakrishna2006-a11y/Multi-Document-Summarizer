from document_vector_store import vector_store
from document_embedding import embedding
from langchain_chroma import Chroma

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="rag_chroma.db",
    collection_name="sample"
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k" : 5
    }
)

result = retriever.invoke("What is langchain?")
print("\n========== RETRIEVED DOCUMENTS ==========")

for i, doc in enumerate(result, start=1):

    print(f"\nDocument {i}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)


print("\n========== RETRIEVAL COMPLETED ==========")