"""
app.py
──────
NotebookLM Replica — Main Streamlit Application
Nunnari Academy | Generative AI & Agentic AI Bootcamp | Capstone Project

3-Panel Layout:
  ┌──────────────┬───────────────────────────┬──────────────┐
  │   SIDEBAR    │        CHAT AREA          │  NOTES PANEL │
  │ PDF Upload   │  st.chat_message          │  Saved Notes │
  │ Doc List     │  st.chat_input            │  Download    │
  │ Settings     │  Sources + Save Note btn  │  Delete      │
  └──────────────┴───────────────────────────┴──────────────┘

Run: streamlit run app.py
"""

import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NotebookLM Replica",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean up the default Streamlit padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Source expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #888;
    }

    /* Sidebar header */
    [data-testid="stSidebar"] h2 {
        margin-top: 0;
    }

    /* Notes panel card styling */
    .note-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid #6c63ff;
    }

    /* Divider between panels */
    .panel-divider {
        border-left: 1px solid #e0e0e0;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Import Components ─────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.chat import render_chat
from components.notes import render_notes_panel

# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    """
    Main application entry point.
    Renders the 3-panel layout: Sidebar | Chat | Notes
    """

    # ── 1. SIDEBAR (Left Panel) ──────────────────────────────────────────────
    # Returns settings selected by the user
    settings = render_sidebar()

    selected_files      = settings["selected_files"]
    web_search_enabled  = settings["web_search_enabled"]

    # ── 2. MAIN CONTENT — Split into Chat + Notes ────────────────────────────
    # Streamlit columns create the 2/3 + 1/3 split for chat and notes
    chat_col, notes_col = st.columns([1.8, 1], gap="medium")

    # ── 3. CHAT AREA (Center Panel) ──────────────────────────────────────────
    with chat_col:
        render_chat(
            selected_files=selected_files,
            web_search_enabled=web_search_enabled,
        )

    # ── 4. NOTES PANEL (Right Panel) ─────────────────────────────────────────
    with notes_col:
        st.markdown('<div class="panel-divider">', unsafe_allow_html=True)
        render_notes_panel()
        st.markdown('</div>', unsafe_allow_html=True)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
