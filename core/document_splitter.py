from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from document_loader import load_document

file_paths = [
    r"C:\Users\patak\OneDrive\Documents\GitHub\Multi doc summarizer\LangChain Notes.pdf",
    r"C:\Users\patak\OneDrive\Documents\GitHub\Multi doc summarizer\Module-4_Inheritance_and_Interfaces_JAVA.pdf"
]

docs = load_document(file_paths)

print(f"\nTotal pages loaded: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100,
    separators=[
        "\n\n", "\n", " ", ""
    ]
)

chunks = splitter.split_documents(docs)
print(f"Total document:{len(docs)}")
print(f"Total chunks:{len(chunks)}")