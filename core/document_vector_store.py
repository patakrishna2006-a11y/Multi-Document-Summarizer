from langchain_classic.vectorstores import Chroma
from document_embedding import embedding
from document_splitter import chunks

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="rag_chroma.db",
    collection_name="sample"
)

vector_store.add_documents(chunks)

print("Vector store generated successfully")