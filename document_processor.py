"""
core/document_processor.py
──────────────────────────
Handles PDF uploading, loading, and chunking.
Concepts from Day 2: Document Loaders + Text Splitters.
"""

import os
import shutil
from datetime import datetime
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


def save_uploaded_pdf(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to the local uploads directory.
    Returns the full file path.
    """
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF using PyPDFLoader and split it into overlapping chunks.
    Each chunk is enriched with metadata: filename, page_number, upload_date.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        A list of LangChain Document objects (chunks) with metadata.
    """
    filename = os.path.basename(file_path)
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Step 1: Load the PDF — each page becomes a Document
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Step 2: Chunk using RecursiveCharacterTextSplitter
    # chunk_size=1000 characters, overlap=200 ensures context continuity
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # Split order: paragraph → line → word
    )

    chunks = splitter.split_documents(pages)

    # Step 3: Attach rich metadata to every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "filename": filename,
            "page_number": chunk.metadata.get("page", 0) + 1,  # 1-indexed
            "upload_date": upload_date,
            "chunk_index": i,
            "source": file_path,
        })

    return chunks


def get_uploaded_documents() -> List[str]:
    """
    Return a list of PDF filenames currently in the uploads directory.
    """
    if not os.path.exists(config.UPLOAD_DIR):
        return []

    return [
        f for f in os.listdir(config.UPLOAD_DIR)
        if f.lower().endswith(".pdf")
    ]


def delete_document(filename: str) -> bool:
    """
    Delete a PDF from the uploads directory.
    Returns True if successful, False otherwise.
    """
    file_path = os.path.join(config.UPLOAD_DIR, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        print(f"Error deleting {filename}: {e}")
    return False
