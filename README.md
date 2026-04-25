# 📄 DocChat — Production-Grade Conversational RAG with Citations

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1.x-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-Gemma2-F55036?style=for-the-badge&logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" />
</p>

<p align="center">
  <i>Upload PDFs. Ask questions. Get answers — with sources, page numbers, and full conversation memory.</i>
</p>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Conversational Memory** | History-aware retrieval rewrites every new query in the context of prior turns — no lost context across a long chat |
| 📎 **Source Citations** | Every answer is backed by document name, page number, and a relevant text snippet |
| ⚡ **Fast Inference** | Powered by [Groq](https://groq.com/) running **Gemma 2 9B** — near-instant LLM responses |
| 🗄️ **Smart Caching** | Embeddings and vector stores are cached via Streamlit `@st.cache_resource` — no recomputation on re-runs |
| 📂 **Multi-PDF Support** | Upload and query multiple PDFs simultaneously; citations always trace back to the correct file |
| 🎨 **Professional UI** | Custom IBM Plex typography, styled citation cards, and a clean dark-accent chat interface |
| 🛡️ **Robust Error Handling** | Graceful messages for missing API keys, empty uploads, and retrieval failures |

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io/) |
| **LLM** | [Groq API](https://groq.com/) — Gemma 2 9B (`langchain-groq`) |
| **Orchestration** | [LangChain Classic](https://pypi.org/project/langchain-classic/) — RAG chain builders |
| **Embeddings** | `all-MiniLM-L6-v2` via [HuggingFace](https://huggingface.co/) (`langchain-huggingface`) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) (`langchain-chroma`) |
| **PDF Loading** | `PyPDFLoader` via `langchain-community` |
| **Text Splitting** | `RecursiveCharacterTextSplitter` via `langchain-text-splitters` |
| **Chat History** | `ChatMessageHistory` + `RunnableWithMessageHistory` (`langchain-core`) |
| **Env Management** | `python-dotenv` |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11** or **3.12**
- A free [Groq API key](https://console.groq.com/keys)
- A free [HuggingFace token](https://huggingface.co/settings/tokens) (for downloading the embedding model)

---

### 1 — Clone the Repository

```bash
git clone https://github.com/AmrAldaly/DocChat-RAG.git
cd DocChat-RAG
```

---

### 2 — Create & Activate a Virtual Environment

```bash
# Create
python -m venv .venv

# Activate (macOS / Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

---

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB) from HuggingFace. It is cached locally after that.

---

### 4 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env   # if provided, or create manually
```

Open `.env` and fill in your keys:

```dotenv
# .env

# Required — get yours at https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_api_key_here

# Required — get yours at https://huggingface.co/settings/tokens
HF_TOKEN=hf_your_huggingface_token_here
```

> ⚠️ Never commit `.env` to version control. It is already listed in `.gitignore`.

---

## 💻 Usage

### Run the App

```bash
streamlit run app.py
```

The app opens automatically at **`http://localhost:8501`**.

### Interacting with DocChat

1. **Enter your Groq API key** in the sidebar.
2. **Upload one or more PDF files** using the file uploader.
3. *(Optional)* Set a **Session ID** to keep conversations isolated between topics.
4. **Type your question** in the chat input at the bottom of the page.
5. Read the **answer** and expand the **📎 Sources** section beneath it to see exactly which document and page the answer came from.
6. Use **🗑️ Clear Chat History** in the sidebar to start a fresh conversation at any time.

---

## 🔬 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER UPLOAD                          │
│              (one or more PDF files)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   PDF PROCESSING PIPELINE                   │
│                                                             │
│  PyPDFLoader ──► RecursiveCharacterTextSplitter             │
│                  (chunk_size=600, overlap=80)               │
│                         │                                   │
│                         ▼                                   │
│  HuggingFaceEmbeddings (all-MiniLM-L6-v2)                  │
│                         │                                   │
│                         ▼                                   │
│  ChromaDB Vector Store  ◄── @st.cache_resource              │
│  (cached by MD5 hash of uploaded files)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONVERSATIONAL RAG CHAIN                  │
│                                                             │
│  1. History-Aware Retriever                                 │
│     └─ Rewrites the user query using prior chat turns       │
│        to ensure standalone, context-free retrieval         │
│                         │                                   │
│  2. Chroma Retriever (top-k=4 most similar chunks)          │
│                         │                                   │
│  3. Stuff-Documents Chain                                   │
│     └─ Injects retrieved chunks into the QA prompt         │
│        and calls Groq (Gemma 2 9B) for the final answer     │
│                         │                                   │
│  4. RunnableWithMessageHistory                              │
│     └─ Persists per-session chat history in                 │
│        st.session_state across turns                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                       │
│                                                             │
│  st.chat_message  ──  answer rendered in chat bubble       │
│  render_citations ──  source file · page · snippet card    │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Caching by file hash** — the vector store is rebuilt only when the uploaded files actually change, identified by their MD5 digest. Re-uploading the same files is instant.
- **History-aware retrieval** — a dedicated LLM call reformulates the user's question before retrieval, so follow-up questions like *"Can you expand on that?"* retrieve relevant chunks rather than failing.
- **Source deduplication** — the citations renderer deduplicates by `(filename, page)` so the same page is never listed twice even if multiple chunks from it were retrieved.
- **Modular helpers** — `build_vectorstore`, `build_rag_chain`, `render_citations`, and `get_session_history` are kept small and independently testable.

---

## 🗺️ Future Enhancements

- [ ] 📁 **Additional file types** — support for `.docx`, `.txt`, `.csv`, and `.md` alongside PDFs
- [ ] 🖥️ **Local LLM integration** — swap Groq for [Ollama](https://ollama.com/) to run fully offline with models like Llama 3 or Mistral
- [ ] 🔎 **Hybrid search** — combine dense vector similarity with BM25 keyword search (reciprocal rank fusion) for improved retrieval precision
- [ ] 💾 **Persistent vector store** — save the ChromaDB index to disk so documents survive app restarts
- [ ] 🔐 **Authentication** — add a login layer for multi-user deployments
- [ ] 📊 **Retrieval analytics** — log queries, retrieved chunks, and latency to a simple dashboard
- [ ] 🌐 **Multi-language support** — swap the embedding model for a multilingual variant (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)
- [ ] ☁️ **Cloud deployment** — one-click deploy to Streamlit Community Cloud or Docker + Fly.io

---

## 📁 Project Structure

```
DocChat-RAG/
├── app.py               # Main Streamlit application
├── requirements.txt     # All pinned Python dependencies
├── .env                 # API keys (never commit this)
├── .env.example         # Safe template to share with collaborators
├── .gitignore
└── README.md
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a [GitHub Issue](https://github.com/AmrAldaly/DocChat-RAG/issues) or submit a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'feat: add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using LangChain · Groq · ChromaDB · Streamlit
</p>
