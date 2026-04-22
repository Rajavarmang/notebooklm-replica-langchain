"""
core/rag_chain.py
─────────────────
Implements the RAG (Retrieval-Augmented Generation) pipeline.
Concepts from Day 3: RAG = Retrieve → Augment → Generate.
"""

from typing import List, Optional, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config
from core.prompts import RAG_PROMPT, WEB_SEARCH_PROMPT
from core.vector_store import search_documents


def get_llm() -> ChatOllama:
    """
    Return a ChatOllama LLM instance using llama3.2:3b.
    Runs completely locally — no API key needed.
    """
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.3,       # Slightly creative but mostly factual
        num_predict=1024,      # Max tokens in response
    )


def format_docs_with_citations(docs: List[Document]) -> str:
    """
    Format retrieved document chunks into a context string.
    Includes filename and page number for citations.
    """
    if not docs:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page_number", "?")
        formatted.append(
            f"[{i}] Source: {filename}, Page {page}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


def run_rag_chain(
    question: str,
    selected_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB (filtered by selected files)
    2. Format chunks as context
    3. Generate answer using the LLM + RAG prompt

    Returns:
        dict with 'answer' (str) and 'sources' (list of Document)
    """
    # Step 1: Retrieve top-k relevant chunks
    retrieved_docs = search_documents(
        query=question,
        selected_files=selected_files,
        k=config.TOP_K_RESULTS,
    )

    if not retrieved_docs:
        return {
            "answer": "⚠️ No relevant content found in the selected documents. Try selecting more documents or rephrasing your question.",
            "sources": [],
        }

    # Step 2: Format context with citations
    context = format_docs_with_citations(retrieved_docs)

    # Step 3: Build and invoke the RAG chain
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "answer": answer,
        "sources": retrieved_docs,
    }


def run_rag_with_web(
    question: str,
    web_results: str,
    selected_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run RAG combined with web search results.
    Used when the Web Search toggle is enabled.
    """
    # Retrieve document context (may be empty if no docs selected)
    retrieved_docs = search_documents(
        query=question,
        selected_files=selected_files,
        k=config.TOP_K_RESULTS,
    )

    context = format_docs_with_citations(retrieved_docs) if retrieved_docs else "No documents selected."

    llm = get_llm()
    chain = WEB_SEARCH_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "web_results": web_results,
        "question": question,
    })

    return {
        "answer": answer,
        "sources": retrieved_docs,
    }
