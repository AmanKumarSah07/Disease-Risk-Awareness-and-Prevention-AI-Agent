from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
RAW_DIR = Path("data/raw")

def _get_embeddings():
    # Free local embeddings (downloads model once)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_pdfs(paths: List[str] | None = None) -> int:
    """
    Ingest PDFs from data/raw (default) or provided file paths.
    Returns number of chunks added to the vector store.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if paths is None:
        pdfs = sorted([p for p in RAW_DIR.glob("*.pdf")])
    else:
        pdfs = [Path(p) for p in paths]

    if not pdfs:
        print("No PDFs found to ingest. Put PDFs in data/raw/*.pdf")
        return 0

    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma(
        collection_name="medical_knowledge",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=_get_embeddings(),
    )

    vectordb.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks into {CHROMA_PERSIST_DIR}")
    return len(chunks)

if __name__ == "__main__":
    ingest_pdfs()