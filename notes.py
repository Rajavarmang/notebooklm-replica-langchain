"""
components/notes.py
───────────────────
Renders the right-side Notes Panel with:
  - Saved notes as collapsible cards
  - Delete individual note
  - Download all notes as a single .md file
"""

import streamlit as st

from utils.helpers import get_all_notes, delete_note, download_all_notes
import config


def render_notes_panel():
    """
    Render the Notes Panel showing all saved markdown notes.
    """
    st.markdown("## 🗒️ Saved Notes")

    notes = get_all_notes()

    # ── Header Row ───────────────────────────────────────────────────────────
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.caption(f"{len(notes)} note(s) saved")
    with col2:
        if notes:
            # Download All Notes button
            all_notes_md = download_all_notes()
            st.download_button(
                label="⬇️ Download All",
                data=all_notes_md,
                file_name="notebook_lm_notes.md",
                mime="text/markdown",
                use_container_width=True,
            )

    st.divider()

    # ── Empty State ──────────────────────────────────────────────────────────
    if not notes:
        st.markdown(
            """
            <div style="text-align:center; padding: 2rem; color: #888;">
                <div style="font-size: 2rem;">📝</div>
                <p>No notes yet.</p>
                <p style="font-size: 0.85rem;">
                    Chat with your documents and click<br/>
                    <strong>💾 Save as Note</strong> to save important answers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Note Cards ───────────────────────────────────────────────────────────
    display_limit = config.MAX_NOTES_DISPLAY
    visible_notes = notes[:display_limit]

    for note in visible_notes:
        _render_note_card(note)

    if len(notes) > display_limit:
        st.caption(f"Showing {display_limit} of {len(notes)} notes. Download all to see more.")


def _render_note_card(note: dict):
    """
    Render a single note as a collapsible card with delete option.
    """
    with st.expander(
        label=f"📄 {note['title'][:45]}{'...' if len(note['title']) > 45 else ''}",
        expanded=False,
    ):
        # Metadata row
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.caption(f"🕐 {note['created_at']}")
        with col2:
            # Delete button
            delete_key = f"del_note_{note['filename']}"
            if st.button("🗑️ Delete", key=delete_key, use_container_width=True):
                if delete_note(note["filename"]):
                    st.success("Note deleted!")
                    st.rerun()
                else:
                    st.error("Failed to delete note.")

        st.divider()

        # Note content (rendered as markdown)
        st.markdown(note["content"])

        # Individual download
        st.download_button(
            label="⬇️ Download Note",
            data=note["content"],
            file_name=note["filename"],
            mime="text/markdown",
            key=f"dl_note_{note['filename']}",
            use_container_width=True,
        )
