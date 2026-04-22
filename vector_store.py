"""
core/vector_store.py
────────────────────
Manages the ChromaDB vector store — adding documents, querying with filters.
Concepts from Day 3: Vector Databases + Embeddings.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config


def get_embeddings() -> OllamaEmbeddings:
    """
    Return an Ollama embeddings instance using nomic-embed-text.
    This runs 100% locally — no API key required.
    """
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def get_vector_store() -> Chroma:
    """
    Return a persistent ChromaDB vector store.
    Data is saved to disk at CHROMA_DB_DIR so it survives app restarts.
    """
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DB_DIR,
    )


def add_documents_to_store(chunks: List[Document]) -> int:
    """
    Add document chunks to ChromaDB.
    Generates unique IDs based on filename + chunk index to avoid duplicates.

    Args:
        chunks: List of Document chunks with metadata.

    Returns:
        Number of chunks successfully added.
    """
    if not chunks:
        return 0

    vector_store = get_vector_store()

    # Build unique IDs to prevent duplicate entries on re-upload
    ids = [
        f"{chunk.metadata['filename']}_chunk_{chunk.metadata['chunk_index']}"
        for chunk in chunks
    ]

    # Add to ChromaDB (embeddings are generated automatically)
    vector_store.add_documents(documents=chunks, ids=ids)

    return len(chunks)


def search_documents(
    query: str,
    selected_files: Optional[List[str]] = None,
    k: int = config.TOP_K_RESULTS,
) -> List[Document]:
    """
    Search ChromaDB for the most relevant chunks.
    Optionally filter by selected filenames using ChromaDB's where clause.

    Args:
        query: User's question or search query.
        selected_files: List of filenames to restrict search to.
        k: Number of top results to return.

    Returns:
        List of the most relevant Document chunks.
    """
    vector_store = get_vector_store()

    # Build metadata filter for selected documents only
    # This is the Day 3 concept: metadata filtering in vector search
    search_kwargs = {"k": k}

    if selected_files and len(selected_files) > 0:
        if len(selected_files) == 1:
            # Single document filter
            search_kwargs["filter"] = {"filename": selected_files[0]}
        else:
            # Multi-document filter using $or operator
            search_kwargs["filter"] = {
                "$or": [{"filename": f} for f in selected_files]
            }

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    results = retriever.invoke(query)
    return results


def get_all_indexed_filenames() -> List[str]:
    """
    Return a list of all unique filenames indexed in ChromaDB.
    Useful for showing which documents are already embedded.
    """
    try:
        vector_store = get_vector_store()
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])
        filenames = list({
            meta.get("filename", "unknown")
            for meta in results.get("metadatas", [])
            if meta
        })
        return sorted(filenames)
    except Exception:
        return []


def delete_document_from_store(filename: str) -> bool:
    """
    Remove all chunks belonging to a specific filename from ChromaDB.
    """
    try:
        vector_store = get_vector_store()
        collection = vector_store._collection
        collection.delete(where={"filename": filename})
        return True
    except Exception as e:
        print(f"Error deleting {filename} from vector store: {e}")
        return False
