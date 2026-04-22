"""
core/agents.py
──────────────
Defines the three tools used by the ReAct agent:
  1. document_search — RAG over uploaded PDFs
  2. web_search      — Tavily internet search
  3. save_note       — Save answer as markdown note

Concepts from Day 4: Agents + Tool Calling.
"""

import os
from datetime import datetime
from typing import List, Optional

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

import config
from core.rag_chain import run_rag_chain, get_llm
from core.vector_store import search_documents
from core.prompts import NOTE_SUMMARY_PROMPT, AGENT_SYSTEM_PROMPT

# ── Shared state passed at runtime ────────────────────────────────────────────
# We use a mutable container so tools can access session state at call time
_tool_context = {
    "selected_files": [],
    "web_search_enabled": False,
}


def set_tool_context(selected_files: List[str], web_search_enabled: bool):
    """Update shared tool context before each agent invocation."""
    _tool_context["selected_files"] = selected_files
    _tool_context["web_search_enabled"] = web_search_enabled


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1: Document Search
# ─────────────────────────────────────────────────────────────────────────────
@tool
def document_search(query: str) -> str:
    """
    Search through the user's uploaded PDF documents using RAG.
    Use this tool for any question that might be answered by the uploaded documents.
    Returns relevant excerpts with source citations (filename + page number).
    """
    selected = _tool_context.get("selected_files", [])

    result = run_rag_chain(question=query, selected_files=selected)
    answer = result["answer"]
    sources = result["sources"]

    # Format sources for the agent to see
    if sources:
        source_lines = [
            f"  - {s.metadata.get('filename', '?')}, Page {s.metadata.get('page_number', '?')}"
            for s in sources
        ]
        source_str = "\n".join(source_lines)
        return f"{answer}\n\nSources:\n{source_str}"

    return answer


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2: Web Search (Tavily)
# ─────────────────────────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the internet for current information using Tavily.
    Use this tool ONLY when the user explicitly asks for web/online information
    or asks about recent events not in the documents.
    Returns a summary of web search results.
    """
    if not _tool_context.get("web_search_enabled", False):
        return "Web search is currently disabled. The user can enable it in the sidebar."

    if not config.TAVILY_API_KEY:
        return "Tavily API key is not configured. Add TAVILY_API_KEY to your .env file."

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        search_tool = TavilySearchResults(
            api_key=config.TAVILY_API_KEY,
            max_results=4,
        )
        results = search_tool.invoke(query)

        # Format results
        if isinstance(results, list):
            formatted = []
            for r in results:
                title = r.get("title", "Result")
                url = r.get("url", "")
                content = r.get("content", "")[:300]
                formatted.append(f"**{title}**\n{content}\n🔗 {url}")
            return "\n\n---\n\n".join(formatted)

        return str(results)

    except Exception as e:
        return f"Web search error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3: Save Note
# ─────────────────────────────────────────────────────────────────────────────
@tool
def save_note(content: str) -> str:
    """
    Save important information or an answer as a markdown note.
    Use this when the user asks to save, remember, or note something.
    The content will be summarized and saved as a .md file.
    """
    try:
        llm = get_llm()
        from langchain_core.output_parsers import StrOutputParser

        # Summarize the content into a clean note
        chain = NOTE_SUMMARY_PROMPT | llm | StrOutputParser()
        note_content = chain.invoke({"answer": content})

        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}.md"
        file_path = os.path.join(config.NOTES_DIR, filename)

        # Save the note to disk
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Note — {datetime.now().strftime('%B %d, %Y %H:%M')}\n\n")
            f.write(note_content)

        return f"✅ Note saved successfully as `{filename}`"

    except Exception as e:
        return f"Failed to save note: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Agent Builder
# ─────────────────────────────────────────────────────────────────────────────
def get_tools():
    """Return the list of available tools."""
    return [document_search, web_search, save_note]


def build_agent_executor(
    selected_files: List[str],
    web_search_enabled: bool,
) -> AgentExecutor:
    """
    Build a ReAct agent with the three tools.
    ReAct = Reason + Act loop — well-suited for llama3.2:3b.

    Args:
        selected_files: Filenames to restrict document search to.
        web_search_enabled: Whether Tavily web search is allowed.

    Returns:
        An AgentExecutor ready to handle queries.
    """
    # Update shared context so tools know what files/mode to use
    set_tool_context(selected_files, web_search_enabled)

    llm = get_llm()
    tools = get_tools()

    # ReAct prompt template
    react_prompt = PromptTemplate.from_template("""
{system}

You have access to the following tools:
{tools}

Use the following format EXACTLY:
Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt.partial(system=AGENT_SYSTEM_PROMPT),
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,          # Prevent infinite loops
        handle_parsing_errors=True, # Graceful error recovery
        return_intermediate_steps=False,
    )

    return executor
