"""
utils/helpers.py
────────────────
Shared helper utilities used across components.
"""

import os
import glob
from datetime import datetime
from typing import List, Dict

import config


def get_all_notes() -> List[Dict]:
    """
    Read all saved notes from the notes directory.
    Returns a list of dicts with filename, content, and created_at.
    """
    notes = []
    pattern = os.path.join(config.NOTES_DIR, "*.md")

    for filepath in sorted(glob.glob(pattern), reverse=True):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from first heading (# Title)
            lines = content.split("\n")
            title = "Untitled Note"
            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Get file creation time
            created_at = datetime.fromtimestamp(
                os.path.getmtime(filepath)
            ).strftime("%b %d, %Y %H:%M")

            notes.append({
                "filename": filename,
                "filepath": filepath,
                "title": title,
                "content": content,
                "created_at": created_at,
            })
        except Exception:
            continue

    return notes


def save_note_directly(content: str, title: str = "Note") -> str:
    """
    Save a note directly (bypassing the agent tool).
    Used when user clicks 'Save as Note' button in chat.

    Returns:
        The filename of the saved note.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"note_{timestamp}.md"
    filepath = os.path.join(config.NOTES_DIR, filename)

    note_body = f"# {title} — {datetime.now().strftime('%B %d, %Y %H:%M')}\n\n{content}\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(note_body)

    return filename


def download_all_notes() -> str:
    """
    Concatenate all notes into a single markdown string for download.
    """
    notes = get_all_notes()
    if not notes:
        return "# No Notes Yet\n\nStart a conversation and save responses as notes!"

    combined = f"# NotebookLM — All Notes\nExported: {datetime.now().strftime('%B %d, %Y')}\n\n---\n\n"

    for note in notes:
        combined += f"{note['content']}\n\n---\n\n"

    return combined


def delete_note(filename: str) -> bool:
    """Delete a single note by filename."""
    filepath = os.path.join(config.NOTES_DIR, filename)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
    except Exception:
        pass
    return False


def format_file_size(bytes_size: int) -> str:
    """Convert bytes to a human-readable size string."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.1f} KB"
    else:
        return f"{bytes_size / (1024**2):.1f} MB"


def get_file_info(filename: str) -> Dict:
    """Return metadata about an uploaded file."""
    filepath = os.path.join(config.UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        return {}

    stat = os.stat(filepath)
    return {
        "filename": filename,
        "size": format_file_size(stat.st_size),
        "uploaded_at": datetime.fromtimestamp(stat.st_mtime).strftime("%b %d, %Y"),
    }


def check_ollama_running() -> bool:
    """Check if Ollama is reachable at the configured base URL."""
    try:
        import urllib.request
        url = f"{config.OLLAMA_BASE_URL}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def check_tavily_configured() -> bool:
    """Check if Tavily API key is set."""
    return bool(config.TAVILY_API_KEY and config.TAVILY_API_KEY.startswith("tvly-"))
