
from langchain_ollama import OllamaEmbeddings

MODEL_NAME = "embeddinggemma-2:latest"

try:
    embedding = OllamaEmbeddings(
        model=MODEL_NAME
    )

    # Validate that the model is available and working
    test_embedding = embedding.embed_query("Test embedding")

    if not test_embedding:
        raise ValueError("Embedding model returned an empty vector")

    print(f"Embedding model initialized: {MODEL_NAME}")

except Exception as e:
    raise RuntimeError(
        f"Failed to initialize embedding model: {e}"
    ) from e
