
import os
from langchain_classic.vectorstores import Chroma
from document_embedding import embedding
from document_splitter import chunks


PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY",
    "rag_chroma.db"
)

COLLECTION_NAME = "sample"


def create_vector_store():

    if not chunks:
        raise ValueError(
            "No document chunks available for indexing."
        )

    try:
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )

        vector_store.add_documents(
            documents=chunks,
            ids=[
                f"document_chunk_{i}"
                for i in range(len(chunks))
            ]
        )

        print(
            f"Vector store generated successfully. "
            f"Total chunks: {len(chunks)}"
        )

        return vector_store

    except Exception as e:
        raise RuntimeError(
            f"Vector store creation failed: {e}"
        ) from e


vector_store = create_vector_store()
