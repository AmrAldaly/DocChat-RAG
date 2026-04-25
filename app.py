"""
Production-Quality Conversational RAG System
============================================
A Streamlit application for conversational Q&A over uploaded PDFs,
featuring citations, chat history, caching, and professional UX.
"""

import os
import tempfile
import hashlib
import streamlit as st

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LangChain 1.x import map
# ---------------------------------------------------------------------------
# In LangChain 1.x the monolithic `langchain` package was refactored into
# several independent pip packages.  The "classic" chain-builder helpers
# (create_history_aware_retriever, create_retrieval_chain,
#  create_stuff_documents_chain) now live in `langchain-classic`.
#
#   pip install langchain-classic     → chain builder helpers
#   pip install langchain-core        → prompts, runnables, base interfaces
#   pip install langchain-community   → loaders, chat message histories
#   pip install langchain-chroma      → Chroma vector store
#   pip install langchain-huggingface → HuggingFaceEmbeddings
#   pip install langchain-groq        → ChatGroq LLM
#   pip install langchain-text-splitters → RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------

# Classic chain builders (pip install langchain-classic)
from langchain_classic.chains import (
    create_history_aware_retriever,   # rewrites the user query using chat history
    create_retrieval_chain,           # wires retriever + QA chain together
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,     # "stuff" retrieved docs into a single prompt
)

# Integrations — each installed as its own pip package
from langchain_chroma import Chroma                                          # langchain-chroma
from langchain_community.chat_message_histories import ChatMessageHistory    # langchain-community
from langchain_community.document_loaders import PyPDFLoader                # langchain-community
from langchain_core.chat_history import BaseChatMessageHistory              # langchain-core
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # langchain-core
from langchain_core.runnables.history import RunnableWithMessageHistory     # langchain-core
from langchain_groq import ChatGroq                                         # langchain-groq
from langchain_huggingface import HuggingFaceEmbeddings                    # langchain-huggingface
from langchain_text_splitters import RecursiveCharacterTextSplitter         # langchain-text-splitters

# ---------------------------------------------------------------------------
# Environment & Page Config
# ---------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="DocChat — RAG Assistant",
    page_icon="📄",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, professional dark-ish theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: -0.5px;
}

.source-card {
    background: #f8f9fa;
    border-left: 3px solid #4f6df5;
    border-radius: 4px;
    padding: 10px 14px;
    margin-top: 6px;
    font-size: 0.82rem;
    color: #444;
    font-family: 'IBM Plex Mono', monospace;
}

.source-card strong {
    color: #4f6df5;
}

.source-header {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #888;
    margin-top: 12px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_SESSION = "default_session"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
RETRIEVER_TOP_K = 4
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load HuggingFace embeddings once and reuse across sessions."""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner="🔍 Processing documents…")
def build_vectorstore(file_hashes: tuple, _file_bytes_list: list, file_names: list) -> Chroma:
    """
    Build (or retrieve from cache) a Chroma vector store for the given files.

    Args:
        file_hashes:     Tuple of MD5 hashes — used as the cache key.
        _file_bytes_list: Raw bytes for each uploaded PDF (prefixed with _ to
                          skip Streamlit's unhashable-arg check).
        file_names:      Original filenames for source tracking.

    Returns:
        A Chroma retriever-ready vector store.
    """
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_splits = []

    for raw_bytes, name in zip(_file_bytes_list, file_names):
        # Write to a temp file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        finally:
            os.unlink(tmp_path)  # clean up immediately

        # Attach the original filename to every chunk's metadata
        for doc in docs:
            doc.metadata["source_file"] = name

        splits = splitter.split_documents(docs)
        all_splits.extend(splits)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
    return vectorstore


# ---------------------------------------------------------------------------
# Session-State Helpers
# ---------------------------------------------------------------------------

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (or initialise) the ChatMessageHistory for a given session."""
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def init_chat_history() -> None:
    """Ensure the display-level chat log exists in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


# ---------------------------------------------------------------------------
# RAG Chain Construction
# ---------------------------------------------------------------------------

def build_rag_chain(llm: ChatGroq, retriever):
    """
    Assemble the full conversational RAG chain:
      1. History-aware retriever  — rewrites the user query using chat history
      2. Stuff-documents chain    — answers the query with retrieved context
      3. RunnableWithMessageHistory — wires in persistent session history
    """

    # --- Step 1: Contextualise the query against chat history ---
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, which may "
         "reference the history, reformulate a standalone question that "
         "can be understood without the history. "
         "Do NOT answer — only rewrite if needed, otherwise return as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # --- Step 2: Answer with retrieved context ---
    qa_system_prompt = (
        "You are an advanced AI research assistant. Your primary goal is to answer questions based on the provided context. "
        "However, if the context is insufficient but the topic is mentioned, use your internal knowledge to supplement the answer. "
        "\n\n"
        "STRICT RULES:\n"
        "1. If the answer comes from the context, always cite the document/page.\n"
        "2. If you use your own knowledge because the context is incomplete, start that part with 'Based on general knowledge: '.\n"
        "3. If the topic is completely unrelated to the documents, say you couldn't find it.\n"
        "4. Keep the response professional, factual, and within 5-6 sentences.\n\n"
        "CONTEXT:\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- Step 3: Combine into retrieval chain ---
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# ---------------------------------------------------------------------------
# Citation Rendering
# ---------------------------------------------------------------------------

def render_citations(source_docs: list) -> None:
    """
    Display a formatted 'Sources' section below the assistant's answer.

    Args:
        source_docs: List of LangChain Document objects returned by the retriever.
    """
    if not source_docs:
        return

    st.markdown('<div class="source-header">📎 Sources</div>', unsafe_allow_html=True)

    seen = set()
    for doc in source_docs:
        meta = doc.metadata
        file_name = meta.get("source_file", meta.get("source", "Unknown document"))
        page = meta.get("page", None)
        snippet = doc.page_content.strip().replace("\n", " ")[:220]

        # Deduplicate identical (file, page) pairs
        key = (file_name, page)
        if key in seen:
            continue
        seen.add(key)

        page_label = f" · p. {page + 1}" if page is not None else ""
        st.markdown(
            f'<div class="source-card">'
            f'<strong>{file_name}{page_label}</strong><br>'
            f'{snippet}…'
            f'</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# File Utilities
# ---------------------------------------------------------------------------

def compute_file_hash(data: bytes) -> str:
    """Return the MD5 hex digest of raw bytes — used as a cache key."""
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📄 DocChat")
    st.caption("Upload PDFs and ask questions — answers come with citations.")

    # --- Sidebar: configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_…")
        session_id = st.text_input("Session ID", value=DEFAULT_SESSION)
        st.divider()
        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents to chat with.",
        )
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if "store" in st.session_state:
                st.session_state.store.pop(session_id, None)
            st.rerun()

    init_chat_history()

    # --- Guard: API key required ---
    if not api_key:
        st.info("👈 Enter your Groq API key in the sidebar to get started.")
        return

    # --- Guard: Files required ---
    if not uploaded_files:
        st.info("👈 Upload at least one PDF to begin chatting.")
        return

    # --- Build vector store (cached by file content) ---
    try:
        file_bytes_list = [f.getvalue() for f in uploaded_files]
        file_names = [f.name for f in uploaded_files]
        file_hashes = tuple(compute_file_hash(b) for b in file_bytes_list)

        vectorstore = build_vectorstore(file_hashes, file_bytes_list, file_names)
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    except Exception as e:
        st.error(f"⚠️ Failed to process documents: {e}")
        return

    # --- Initialise LLM & RAG chain ---
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name=LLM_MODEL)
        rag_chain = build_rag_chain(llm, retriever)
    except Exception as e:
        st.error(f"⚠️ Failed to initialise LLM: {e}")
        return

    # --- Render existing conversation ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_citations(msg["sources"])

    # --- Handle new user input ---
    user_input = st.chat_input("Ask a question about your documents…")
    if not user_input:
        return

    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                answer = response.get("answer", "I couldn't generate a response.")
                source_docs = response.get("context", [])
            except Exception as e:
                answer = f"⚠️ An error occurred: {e}"
                source_docs = []

        st.markdown(answer)
        render_citations(source_docs)

    # Persist to display-level history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_docs,
    })


if __name__ == "__main__":
    main()
