"""
core/prompts.py
───────────────
All prompt templates used across the application.
Centralising prompts here makes them easy to refine and version-control.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# ── RAG Answer Prompt ─────────────────────────────────────────────────────────
# Used when answering questions grounded in retrieved document chunks.
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful research assistant inside a NotebookLM-style application.
Your job is to answer the user's question using ONLY the provided document context.

Rules:
- Base your answer strictly on the context below.
- If the answer is not in the context, say "I couldn't find this in the selected documents."
- Always cite your sources at the end in the format: [Source: filename, Page X]
- Be concise, accurate, and well-structured.

Context:
{context}

Question: {question}

Answer (with citations):
""")

# ── Web Search Answer Prompt ──────────────────────────────────────────────────
# Used when answering with web search results combined with doc context.
WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful research assistant.
Use the information from BOTH the document context AND the web search results to answer the question.

Document Context:
{context}

Web Search Results:
{web_results}

Question: {question}

Answer (mention whether info came from documents or web):
""")

# ── Note Summarization Prompt ─────────────────────────────────────────────────
# Used by the Save Note tool to compress an answer into a compact note.
NOTE_SUMMARY_PROMPT = PromptTemplate.from_template("""
Summarize the following answer into a concise, well-structured markdown note.
Include a short title, key points as bullet points, and any citations.

Answer:
{answer}

Markdown Note:
""")

# ── Intent Classification Prompt ──────────────────────────────────────────────
# Used by LangGraph to route the query to the right node.
INTENT_PROMPT = PromptTemplate.from_template("""
Classify the user's intent into one of these categories:
- "document_search": The user is asking a question that can be answered from uploaded documents.
- "web_search": The user explicitly wants web/internet information or asks about recent events.
- "save_note": The user wants to save something or create a note.
- "general": General conversation not needing documents or web.

User message: {query}

Respond with ONLY one of: document_search, web_search, save_note, general
Intent:
""")

# ── ReAct Agent System Prompt ─────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.
You help users explore their uploaded documents and find information.

You have access to these tools:
1. document_search — search through uploaded PDF documents
2. web_search — search the internet (only when user enables it)
3. save_note — save an important answer as a markdown note

Always use document_search first for document-related questions.
Provide clear, well-structured answers with citations when available.
"""
