"""
components/sidebar.py
─────────────────────
Renders the left sidebar with:
  - PDF uploader
  - Document list with checkboxes
  - Web search toggle
  - LangGraph diagram
  - System status indicators
"""

import os
import streamlit as st

import config
from core.document_processor import (
    save_uploaded_pdf,
    load_and_chunk_pdf,
    get_uploaded_documents,
    delete_document,
)
from core.vector_store import (
    add_documents_to_store,
    get_all_indexed_filenames,
    delete_document_from_store,
)
from utils.helpers import get_file_info, check_ollama_running, check_tavily_configured


def render_sidebar() -> dict:
    """
    Render the full sidebar and return the current UI settings.

    Returns:
        dict with keys:
          - selected_files: List[str] of checked filenames
          - web_search_enabled: bool
    """
    with st.sidebar:
        # ── Header ──────────────────────────────────────────────────────────
        st.markdown("## 📓 NotebookLM")
        st.caption("Nunnari Academy · Capstone Project")
        st.divider()

        # ── System Status ────────────────────────────────────────────────────
        _render_status_indicators()
        st.divider()

        # ── PDF Upload ───────────────────────────────────────────────────────
        st.markdown("### 📄 Upload Documents")
        uploaded = st.file_uploader(
            label="Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with",
            label_visibility="collapsed",
        )

        if uploaded:
            _process_uploads(uploaded)

        st.divider()

        # ── Document List ────────────────────────────────────────────────────
        st.markdown("### 📚 Your Documents")
        selected_files = _render_document_list()
        st.divider()

        # ── Chat Settings ────────────────────────────────────────────────────
        st.markdown("### ⚙️ Chat Settings")

        web_search_enabled = st.toggle(
            "🌐 Web Search (Tavily)",
            value=False,
            help="Enable to search the internet for answers not in your documents",
        )

        if web_search_enabled and not check_tavily_configured():
            st.warning("⚠️ Add TAVILY_API_KEY to .env to enable web search", icon="⚠️")

        st.divider()

        # ── LangGraph Diagram ────────────────────────────────────────────────
        _render_graph_diagram()

        st.divider()

        # ── Reset Button ─────────────────────────────────────────────────────
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return {
        "selected_files": selected_files,
        "web_search_enabled": web_search_enabled,
    }


def _render_status_indicators():
    """Show green/red indicators for Ollama and Tavily status."""
    st.markdown("**System Status**")

    ollama_ok = check_ollama_running()
    tavily_ok = check_tavily_configured()

    col1, col2 = st.columns(2)
    with col1:
        status = "🟢" if ollama_ok else "🔴"
        st.markdown(f"{status} **Ollama**")
        if not ollama_ok:
            st.caption("Run: ollama serve")
    with col2:
        status = "🟢" if tavily_ok else "🟡"
        st.markdown(f"{status} **Tavily**")
        if not tavily_ok:
            st.caption("Add to .env")


def _process_uploads(uploaded_files):
    """Save and index newly uploaded PDFs."""
    indexed = get_all_indexed_filenames()

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        if filename in indexed:
            st.info(f"✅ Already indexed: **{filename}**")
            continue

        with st.spinner(f"Processing **{filename}**..."):
            try:
                # Save PDF to disk
                file_path = save_uploaded_pdf(uploaded_file)

                # Load and chunk the PDF
                chunks = load_and_chunk_pdf(file_path)

                # Store chunks in ChromaDB
                count = add_documents_to_store(chunks)

                st.success(f"✅ **{filename}** indexed ({count} chunks)")

            except Exception as e:
                st.error(f"❌ Failed to process **{filename}**: {str(e)}")


def _render_document_list() -> list:
    """
    Render checkboxes for all uploaded documents.
    Returns list of selected filenames.
    """
    docs = get_uploaded_documents()

    if not docs:
        st.caption("No documents uploaded yet. Add PDFs above.")
        return []

    selected = []

    for filename in docs:
        col1, col2 = st.columns([0.8, 0.2])

        with col1:
            info = get_file_info(filename)
            checked = st.checkbox(
                label=filename,
                value=True,                # Default: all selected
                key=f"doc_check_{filename}",
                help=f"Size: {info.get('size', '?')} | Uploaded: {info.get('uploaded_at', '?')}",
            )
            if checked:
                selected.append(filename)

        with col2:
            # Delete button
            if st.button("🗑️", key=f"del_{filename}", help=f"Delete {filename}"):
                delete_document(filename)
                delete_document_from_store(filename)
                st.rerun()

    if selected:
        st.caption(f"✅ {len(selected)} of {len(docs)} document(s) selected")
    else:
        st.caption("⚠️ No documents selected")

    return selected


def _render_graph_diagram():
    """Show the LangGraph workflow Mermaid diagram in an expander."""
    with st.expander("🔀 LangGraph Workflow", expanded=False):
        try:
            from core.graph import get_graph_mermaid
            mermaid_str = get_graph_mermaid()
            st.code(mermaid_str, language="text")
            st.caption("Copy into mermaid.live to visualise")
        except Exception as e:
            st.caption(f"Graph not available: {e}")
