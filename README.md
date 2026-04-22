# 📓 NotebookLM Replica
**Nunnari Academy | Generative AI & Agentic AI Bootcamp | Capstone Project**

A full-featured NotebookLM clone built with **Streamlit + LangChain + LangGraph + Ollama + ChromaDB**. Upload PDFs, chat with your documents using RAG, optionally search the web, and save important answers as notes — all running 100% locally.

---

## 🖼️ Screenshots

> *(Add your screenshots here after running the app)*
> - 3-Panel UI (Sidebar + Chat + Notes)
> - LangGraph Workflow Diagram

---

## ✨ Features — Mapped to Bootcamp Days

| Feature | Day | Concept |
|---------|-----|---------|
| Persona prompting + system prompts | Day 1 | Prompt Engineering |
| PDF upload, PyPDFLoader, chunking | Day 2 | Document Loaders |
| ChromaDB embeddings + RAG | Day 3 | Vector DB + RAG |
| Agent with 3 tools (doc/web/note) | Day 4 | Tool Calling |
| LangGraph workflow + routing | Day 5 | Agentic AI |

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit |
| **LLM** | Ollama — `llama3.2:3b` (local, free) |
| **Framework** | LangChain |
| **Vector DB** | ChromaDB (persistent, local) |
| **Embeddings** | Ollama — `nomic-embed-text` |
| **Web Search** | Tavily API (via LangChain) |
| **Orchestration** | LangGraph |
| **File Storage** | Local filesystem |

---

## 📁 Project Structure

```
notebook-lm/
├── app.py                    # Main Streamlit entry point (3-panel layout)
├── config.py                 # All settings & environment variables
├── requirements.txt
├── .env.example
├── components/
│   ├── sidebar.py            # PDF uploader, doc list, settings
│   ├── chat.py               # Chat interface with st.chat_message
│   └── notes.py              # Notes panel with collapsible cards
├── core/
│   ├── prompts.py            # All LLM prompt templates
│   ├── document_processor.py # PDF loading & chunking (Day 2)
│   ├── vector_store.py       # ChromaDB operations (Day 3)
│   ├── rag_chain.py          # RAG pipeline (Day 3)
│   ├── agents.py             # ReAct agent + 3 tools (Day 4)
│   └── graph.py              # LangGraph workflow (Day 5)
├── storage/
│   ├── uploads/              # Saved PDF files
│   ├── chroma_db/            # Persistent vector embeddings
│   └── notes/                # Saved markdown notes
└── utils/
    └── helpers.py            # Shared utility functions
```

---

## ⚙️ Prerequisites

### 1. Install & Start Ollama
```bash
# Download from https://ollama.com
# Then pull required models:
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Verify Ollama is running:
curl http://localhost:11434/api/tags
```

### 2. Get Tavily API Key (Free)
- Sign up at [https://tavily.com](https://tavily.com)
- Copy your API key

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/notebook-lm.git
cd notebook-lm

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY

# 5. Start Ollama (in a separate terminal)
ollama serve

# 6. Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔄 LangGraph Workflow

```
START
  │
  ▼
classify_intent
  │
  ├── document_search ──► retrieve_documents ──► [save_note] ──► END
  │
  ├── web_search ──────► web_search_node ──────────────────────► END
  │
  ├── save_note ───────► retrieve_documents ──► save_note ──────► END
  │
  └── general ─────────► generate_response ────────────────────► END
```

---

## 🧪 Usage Flow

1. **Upload PDFs** — Drag & drop in the sidebar
2. **Select documents** — Check the ones you want to chat with
3. **Ask questions** — Chat in the center panel
4. **Get grounded answers** — With citations (filename + page)
5. **Enable Web Search** — Toggle in sidebar for internet results
6. **Save notes** — Click 💾 Save as Note on any response
7. **Download notes** — Export all notes as a single `.md` file

---

## 🌱 Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
TAVILY_API_KEY=tvly-your-api-key-here
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 🎓 Author

Built for the **Nunnari Academy — Generative AI & Agentic AI Bootcamp** Capstone.

Journey: **Day 1** (Prompts) → **Day 2** (Documents) → **Day 3** (RAG) → **Day 4** (Agents) → **Day 5** (LangGraph) → **Capstone** 🎉
