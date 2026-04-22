"""
core/graph.py
─────────────
LangGraph workflow orchestrating the full NotebookLM pipeline.
Nodes: classify_intent → retrieve_documents / web_search → generate_response → [save_note]
Concepts from Day 5: LangGraph state machines + conditional routing.
"""

from typing import TypedDict, List, Optional, Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

import config
from core.prompts import INTENT_PROMPT
from core.rag_chain import run_rag_chain, run_rag_with_web, get_llm


# ─────────────────────────────────────────────────────────────────────────────
# State Schema — shared across all nodes
# ─────────────────────────────────────────────────────────────────────────────
class NotebookState(TypedDict):
    """The state object passed between every LangGraph node."""
    query: str                        # User's original question
    intent: str                       # Classified intent
    selected_files: List[str]         # Files selected by user in sidebar
    web_search_enabled: bool          # Toggle state from sidebar
    retrieved_context: str            # Formatted doc chunks
    web_results: str                  # Raw web search results
    final_answer: str                 # Generated response
    sources: list                     # Source documents for citations
    save_note_requested: bool         # Whether to save as note


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Classify Intent
# ─────────────────────────────────────────────────────────────────────────────
def classify_intent(state: NotebookState) -> NotebookState:
    """
    Classify the user's query intent using the LLM.
    Routes: document_search | web_search | save_note | general
    """
    llm = get_llm()
    chain = INTENT_PROMPT | llm | StrOutputParser()

    intent = chain.invoke({"query": state["query"]}).strip().lower()

    # Validate — fallback to document_search if unrecognized
    valid_intents = ["document_search", "web_search", "save_note", "general"]
    if intent not in valid_intents:
        intent = "document_search"

    # If web search is disabled, redirect web_search intents to document_search
    if intent == "web_search" and not state.get("web_search_enabled", False):
        intent = "document_search"

    return {**state, "intent": intent}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Retrieve Documents
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_documents(state: NotebookState) -> NotebookState:
    """
    Run RAG retrieval on the selected documents.
    Returns the formatted context and source documents.
    """
    result = run_rag_chain(
        question=state["query"],
        selected_files=state.get("selected_files", []),
    )

    from core.rag_chain import format_docs_with_citations
    context = format_docs_with_citations(result["sources"]) if result["sources"] else ""

    return {
        **state,
        "retrieved_context": context,
        "sources": result["sources"],
        "final_answer": result["answer"],   # Already has answer from RAG chain
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Web Search
# ─────────────────────────────────────────────────────────────────────────────
def web_search_node(state: NotebookState) -> NotebookState:
    """
    Perform a Tavily web search and combine with document context.
    Only runs when web_search_enabled=True.
    """
    query = state["query"]
    web_results = "Web search not available."

    if state.get("web_search_enabled") and config.TAVILY_API_KEY:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            tavily = TavilySearchResults(
                api_key=config.TAVILY_API_KEY,
                max_results=4,
            )
            results = tavily.invoke(query)

            if isinstance(results, list):
                web_results = "\n\n".join([
                    f"{r.get('title', 'Result')}: {r.get('content', '')[:400]}"
                    for r in results
                ])
            else:
                web_results = str(results)

        except Exception as e:
            web_results = f"Web search error: {str(e)}"

    # Generate combined answer
    combined = run_rag_with_web(
        question=query,
        web_results=web_results,
        selected_files=state.get("selected_files", []),
    )

    return {
        **state,
        "web_results": web_results,
        "final_answer": combined["answer"],
        "sources": combined["sources"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: Generate Response (for general queries)
# ─────────────────────────────────────────────────────────────────────────────
def generate_response(state: NotebookState) -> NotebookState:
    """
    Handle general queries that don't need documents or web search.
    Just uses the LLM directly.
    """
    llm = get_llm()
    chain = llm | StrOutputParser()

    answer = chain.invoke(state["query"])

    return {
        **state,
        "final_answer": answer,
        "sources": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: Save Note
# ─────────────────────────────────────────────────────────────────────────────
def save_note_node(state: NotebookState) -> NotebookState:
    """
    Save the final answer as a markdown note to disk.
    Triggered when save_note intent is detected.
    """
    import os
    from datetime import datetime
    from core.prompts import NOTE_SUMMARY_PROMPT

    try:
        llm = get_llm()
        chain = NOTE_SUMMARY_PROMPT | llm | StrOutputParser()
        note_content = chain.invoke({"answer": state.get("final_answer", state["query"])})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}.md"
        file_path = os.path.join(config.NOTES_DIR, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Note — {datetime.now().strftime('%B %d, %Y %H:%M')}\n\n")
            f.write(note_content)

        return {
            **state,
            "final_answer": state.get("final_answer", "") + f"\n\n✅ Note saved as `{filename}`",
        }

    except Exception as e:
        return {**state, "final_answer": state.get("final_answer", "") + f"\n\n⚠️ Note save failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Routing Functions — Conditional Edges
# ─────────────────────────────────────────────────────────────────────────────
def route_after_classify(state: NotebookState) -> str:
    """
    Decide which node to go to after classification.
    """
    intent = state.get("intent", "document_search")

    if intent == "web_search":
        return "web_search"
    elif intent == "save_note":
        return "retrieve_documents"   # Retrieve first, then save
    elif intent == "general":
        return "generate_response"
    else:
        return "retrieve_documents"


def route_after_retrieve(state: NotebookState) -> str:
    """
    After retrieving documents, decide whether to also save a note.
    """
    if state.get("intent") == "save_note":
        return "save_note"
    return END


def route_after_web(state: NotebookState) -> str:
    """After web search, always go to END (no note saving in web path)."""
    return END


# ─────────────────────────────────────────────────────────────────────────────
# Build the LangGraph
# ─────────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """
    Assemble and compile the full LangGraph workflow.

    Flow:
    START → classify_intent
               ↓ (conditional)
       ┌───────────────────────┐
       │   retrieve_documents  │──→ [save_note] → END
       │   web_search          │──────────────→ END
       │   generate_response   │──────────────→ END
       └───────────────────────┘
    """
    graph = StateGraph(NotebookState)

    # Add all nodes
    graph.add_node("classify_intent",    classify_intent)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("web_search",         web_search_node)
    graph.add_node("generate_response",  generate_response)
    graph.add_node("save_note",          save_note_node)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "retrieve_documents": "retrieve_documents",
            "web_search":         "web_search",
            "generate_response":  "generate_response",
        },
    )

    # After retrieve — conditionally save note or end
    graph.add_conditional_edges(
        "retrieve_documents",
        route_after_retrieve,
        {
            "save_note": "save_note",
            END:         END,
        },
    )

    # Web search → always END
    graph.add_edge("web_search",        END)
    graph.add_edge("generate_response", END)
    graph.add_edge("save_note",         END)

    return graph.compile()


def run_graph(
    query: str,
    selected_files: List[str],
    web_search_enabled: bool,
) -> dict:
    """
    Run the compiled LangGraph with the given inputs.

    Returns:
        dict with 'answer' and 'sources'
    """
    graph = build_graph()

    initial_state: NotebookState = {
        "query": query,
        "intent": "",
        "selected_files": selected_files,
        "web_search_enabled": web_search_enabled,
        "retrieved_context": "",
        "web_results": "",
        "final_answer": "",
        "sources": [],
        "save_note_requested": False,
    }

    result = graph.invoke(initial_state)

    return {
        "answer": result.get("final_answer", "No response generated."),
        "sources": result.get("sources", []),
        "intent": result.get("intent", ""),
    }


def get_graph_mermaid() -> str:
    """Return the Mermaid diagram string for visualising the graph."""
    graph = build_graph()
    return graph.get_graph().draw_mermaid()
