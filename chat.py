"""
components/chat.py
──────────────────
Renders the main chat area with:
  - Message history using st.chat_message
  - Chat input using st.chat_input
  - Source citations below each answer
  - 'Save as Note' button on assistant messages
  - Loading spinner while generating response
"""

import streamlit as st
from typing import List

from core.graph import run_graph
from utils.helpers import save_note_directly


def init_chat_state():
    """Initialise session state variables for chat."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


def render_chat(selected_files: List[str], web_search_enabled: bool):
    """
    Render the full chat interface.

    Args:
        selected_files: Filenames to restrict RAG search to.
        web_search_enabled: Whether Tavily web search is on.
    """
    init_chat_state()

    # ── Chat Header ──────────────────────────────────────────────────────────
    st.markdown("## 💬 Chat with Your Documents")

    if not selected_files:
        st.info(
            "📂 **No documents selected.** Upload PDFs in the sidebar and select them to chat.",
            icon="💡",
        )

    if web_search_enabled:
        st.success("🌐 Web Search is ON — answers may include internet results", icon="🌐")

    # ── Message History ──────────────────────────────────────────────────────
    _render_message_history()

    # ── Chat Input ───────────────────────────────────────────────────────────
    user_input = st.chat_input(
        placeholder="Ask a question about your documents...",
        disabled=False,
    )

    if user_input:
        _handle_user_input(user_input, selected_files, web_search_enabled)


def _render_message_history():
    """Render all past messages in the session."""
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]

        with st.chat_message(role):
            st.markdown(msg["content"])

            # Show sources below assistant messages
            if role == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

            # Save as Note button (assistant messages only)
            if role == "assistant":
                _render_save_note_button(msg["content"], i)


def _handle_user_input(
    user_input: str,
    selected_files: List[str],
    web_search_enabled: bool,
):
    """
    Process user input: add to history, run the graph, display response.
    """
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Run the LangGraph workflow
                result = run_graph(
                    query=user_input,
                    selected_files=selected_files,
                    web_search_enabled=web_search_enabled,
                )

                answer = result.get("answer", "I couldn't generate a response.")
                sources = result.get("sources", [])
                intent = result.get("intent", "")

            except Exception as e:
                answer = f"⚠️ An error occurred: {str(e)}\n\nPlease check that Ollama is running (`ollama serve`)."
                sources = []
                intent = "error"

        # Display the answer
        st.markdown(answer)

        # Show intent badge
        if intent:
            intent_icons = {
                "document_search": "📄 Document Search",
                "web_search": "🌐 Web Search",
                "save_note": "💾 Save Note",
                "general": "💬 General",
            }
            st.caption(f"Mode: {intent_icons.get(intent, intent)}")

        # Display sources
        if sources:
            _render_sources(sources)

        # Save as Note button for the latest response
        _render_save_note_button(answer, len(st.session_state.messages))

    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


def _render_sources(sources: list):
    """Render collapsible source citations."""
    if not sources:
        return

    unique_sources = {}
    for doc in sources:
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page_number", "?")
        key = f"{filename} — Page {page}"
        if key not in unique_sources:
            unique_sources[key] = doc.page_content[:200] + "..."

    with st.expander(f"📌 {len(unique_sources)} Source(s)", expanded=False):
        for citation, preview in unique_sources.items():
            st.markdown(f"**{citation}**")
            st.caption(preview)
            st.divider()


def _render_save_note_button(content: str, message_index: int):
    """Render a 'Save as Note' button for an assistant message."""
    button_key = f"save_note_{message_index}_{hash(content[:50])}"

    if st.button("💾 Save as Note", key=button_key, help="Save this response as a markdown note"):
        try:
            # Extract first line as title
            first_line = content.split("\n")[0].strip()[:60]
            title = first_line if first_line else "Saved Note"

            filename = save_note_directly(content=content, title=title)
            st.success(f"✅ Saved as `{filename}`")
        except Exception as e:
            st.error(f"❌ Failed to save note: {e}")
