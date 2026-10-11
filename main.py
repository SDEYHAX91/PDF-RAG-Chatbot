"""Streamlit interface for asking questions about uploaded documents."""

from __future__ import annotations

import hashlib
import io
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import streamlit as st
from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# pyrefly: ignore [missing-import]
from rag_pipeline import NOT_FOUND, build_pipeline, file_to_documents


def apply_dashboard_theme() -> None:
    """Apply the dark emerald dashboard styling used by the project thumbnail."""
    st.markdown(
        """
        <style>
        :root {
            --emerald: #18b66a;
            --emerald-bright: #4ade80;
            --emerald-dark: #087443;
            --surface: #111716;
            --surface-raised: #17201e;
            --border: #2a3834;
            --muted: #9aaba5;
        }

        .stApp {
            color: #f4f7f5;
            background:
                radial-gradient(circle at 85% 0%, rgba(18, 143, 83, 0.20), transparent 32rem),
                radial-gradient(circle at 10% 100%, rgba(5, 95, 54, 0.16), transparent 28rem),
                #090d0c;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101715 0%, #0b100f 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: #c9d4d0;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        .rag-hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.5rem;
            padding: 1.7rem 2rem;
            border: 1px solid #254237;
            border-radius: 18px;
            background: linear-gradient(115deg, rgba(21, 44, 36, .96), rgba(14, 21, 19, .96));
            box-shadow: 0 18px 50px rgba(0, 0, 0, .28);
        }

        .rag-hero::after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            right: -80px;
            top: -150px;
            border-radius: 50%;
            background: rgba(24, 182, 106, .18);
            filter: blur(20px);
        }

        .rag-eyebrow {
            margin-bottom: .45rem;
            color: var(--emerald-bright);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .rag-title {
            margin: 0;
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.25rem);
            font-weight: 850;
            letter-spacing: -.045em;
            line-height: 1;
        }

        .rag-title span { color: var(--emerald-bright); }

        .rag-subtitle {
            max-width: 690px;
            margin: .85rem 0 0;
            color: var(--muted);
            font-size: 1rem;
        }

        [data-testid="stFileUploaderDropzone"],
        [data-testid="stChatMessage"],
        div[data-testid="stAlert"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(20, 29, 27, .90);
        }

        [data-testid="stChatMessage"] {
            margin-bottom: .7rem;
            padding: .35rem .45rem;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"], [data-testid="stChatMessageAvatarAssistant"]) {
            border-left: 3px solid var(--emerald);
        }

        .stButton > button[kind="primary"] {
            border: 1px solid #31ce80;
            background: linear-gradient(135deg, #15975a, #087443);
            color: white;
            font-weight: 750;
            box-shadow: 0 8px 24px rgba(18, 166, 93, .22);
        }

        .stButton > button[kind="primary"]:hover {
            border-color: #65e59d;
            background: linear-gradient(135deg, #1db86c, #09864d);
        }

        .stButton > button[kind="secondary"] {
            border-color: var(--border);
            background: var(--surface-raised);
            color: #dbe5e1;
        }

        [data-testid="stChatInput"] {
            border: 1px solid #315144;
            border-radius: 14px;
            background: #121a18;
            box-shadow: 0 10px 35px rgba(0, 0, 0, .25);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--emerald);
            box-shadow: 0 0 0 2px rgba(24, 182, 106, .14);
        }

        [data-testid="stFileUploaderDropzone"] button {
            border-color: #277e55;
            background: #163326;
            color: #dff8e9;
        }


        /* Sidebar history */
        .history-title {
            display: flex;
            align-items: center;
            gap: .5rem;
            margin: .2rem 0 .8rem;
            font-size: 1.15rem;
            font-weight: 800;
            color: #f4f7f5;
        }

        [data-testid="stSidebar"] .stButton > button {
            text-align: left;
            justify-content: flex-start;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        /* Upload card in the main area */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border) !important;
            border-radius: 14px;
            background: rgba(20, 29, 27, .55);
        }

        .upload-label {
            margin-bottom: .4rem;
            color: #dbe5e1;
            font-weight: 700;
        }

        [data-testid="stSelectbox"] > div > div {
            border-radius: 10px;
            border-color: var(--border);
            background: var(--surface-raised);
        }

        /* No user / assistant icons in the chat */
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid="stChatMessageAvatarCustom"],
        [data-testid="chatAvatarIcon-user"],
        [data-testid="chatAvatarIcon-assistant"] {
            display: none !important;
        }

        h1, h2, h3 { letter-spacing: -.025em; }
        a { color: var(--emerald-bright) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def uploaded_file_to_documents(uploaded_file) -> list[Document]:
    """Extract text from a Streamlit UploadedFile: one Document per PDF page."""
    return file_to_documents(uploaded_file.name, uploaded_file.getvalue())


def file_set_id(uploaded_files) -> str:
    """Return a stable identifier for the currently uploaded file contents."""
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def build_agent(documents: list[Document]):
    """Build a session-local vector index and the deterministic RAG pipeline."""
    return build_pipeline(documents)


# ---------------------------------------------------------------------------
# Conversation history database (SQLite)
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).with_name("chat_history.db")


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def db_init() -> None:
    with _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      REAL NOT NULL
            );
            """
        )


def db_load_conversations() -> list[dict]:
    """Load saved conversations (newest first) with their messages."""
    with _db() as conn:
        convos = conn.execute(
            "SELECT id, title FROM conversations ORDER BY created_at DESC"
        ).fetchall()
        result = []
        for convo_id, title in convos:
            rows = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (convo_id,),
            ).fetchall()
            result.append(
                {
                    "id": convo_id,
                    "title": title,
                    "messages": [{"role": r, "content": c} for r, c in rows],
                }
            )
        return result


def db_save_message(convo: dict, role: str, content: str) -> None:
    """Store a message, creating/updating its conversation row as needed."""
    now = time.time()
    with _db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title = excluded.title",
            (convo["id"], convo["title"], now),
        )
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (convo["id"], role, content, now),
        )


def db_delete_conversation(convo_id: str) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (convo_id,))


def db_delete_all() -> None:
    with _db() as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def stream_answer(pipeline, messages: list) -> str:
    """Retrieve in code, stream the model's answer, then append code-generated sources."""
    question = messages[-1].content
    history = messages[:-1]
    placeholder = st.empty()
    placeholder.markdown("_Searching your documents..._")

    buffer = ""
    try:
        docs, broad = pipeline.retrieve(question, history)
        for token in pipeline.stream(question, history, docs, broad):
            buffer += token
            placeholder.markdown(buffer + "▌")
        answer = buffer.strip() or NOT_FOUND
        if not answer.startswith(NOT_FOUND):
            answer += pipeline.sources_line(docs)
        placeholder.markdown(answer)
        with st.expander("Retrieved excerpts (what the model actually saw)"):
            for d in docs:
                page = d.metadata.get("page")
                st.caption(d.metadata.get("source", "unknown") + (f" - page {page}" if page else ""))
                st.text(d.page_content[:600])
    except Exception as exc:
        answer = f"I couldn't answer that question: {exc}"
        st.exception(exc)
        placeholder.markdown(answer)
    return answer


# ---------------------------------------------------------------------------
# Conversation history helpers
# ---------------------------------------------------------------------------
def _new_conversation() -> dict:
    return {"id": uuid.uuid4().hex[:8], "title": "New conversation", "messages": []}


def init_state() -> None:
    if "conversations" not in st.session_state:
        db_init()
        first = _new_conversation()
        st.session_state.conversations = [first] + db_load_conversations()
        st.session_state.active_id = first["id"]


def get_active_conversation() -> dict:
    for convo in st.session_state.conversations:
        if convo["id"] == st.session_state.active_id:
            return convo
    st.session_state.active_id = st.session_state.conversations[0]["id"]
    return st.session_state.conversations[0]

def start_new_conversation() -> None:
    """Open a fresh conversation (re-uses the active one if it is still empty)."""
    if not get_active_conversation()["messages"]:
        return
    convo = _new_conversation()
    st.session_state.conversations.insert(0, convo)
    st.session_state.active_id = convo["id"]


def select_conversation(convo_id: str) -> None:
    st.session_state.active_id = convo_id


def delete_conversation(convo_id: str) -> None:
    db_delete_conversation(convo_id)
    st.session_state.conversations = [
        c for c in st.session_state.conversations if c["id"] != convo_id
    ]
    if not st.session_state.conversations:
        st.session_state.conversations = [_new_conversation()]
    if st.session_state.active_id == convo_id:
        st.session_state.active_id = st.session_state.conversations[0]["id"]


def delete_all_history() -> None:
    """Remove every saved conversation (database and session) and start fresh."""
    db_delete_all()
    fresh = _new_conversation()
    st.session_state.conversations = [fresh]
    st.session_state.active_id = fresh["id"]


def render_sidebar() -> None:
    with st.sidebar:
        title_col, new_col = st.columns([3, 1])
        with title_col:
            st.markdown('<div class="history-title">History</div>', unsafe_allow_html=True)
        with new_col:
            st.button(
                "",
                icon=":material/add_circle:",
                key="new_chat",
                help="New conversation",
                on_click=start_new_conversation,
                use_container_width=True,
            )

        for convo in st.session_state.conversations:
            is_active = convo["id"] == st.session_state.active_id
            name_col, del_col = st.columns([5, 1])
            with name_col:
                st.button(
                    convo["title"],
                    key=f"open_{convo['id']}",
                    type="primary" if is_active else "secondary",
                    on_click=select_conversation,
                    args=(convo["id"],),
                    use_container_width=True,
                )
            with del_col:
                with st.popover(
                    "",
                    icon=":material/close:",
                    help="Delete conversation",
                    use_container_width=True,
                ):
                    st.caption("Delete this conversation permanently?")
                    st.button(
                        "Yes, delete",
                        key=f"confirm_del_{convo['id']}",
                        type="primary",
                        on_click=delete_conversation,
                        args=(convo["id"],),
                        use_container_width=True,
                    )

        st.divider()
        with st.popover(
            "Delete all history",
            icon=":material/delete_sweep:",
            use_container_width=True,
        ):
            st.caption("This permanently deletes every saved conversation.")
            st.button(
                "Yes, delete everything",
                key="confirm_delete_all",
                type="primary",
                on_click=delete_all_history,
                use_container_width=True,
            )


def to_langchain_history(messages: list[dict]) -> list:
    """Convert stored chat messages into LangChain messages for context."""
    history = []
    for item in messages:
        if item["role"] == "user":
            history.append(HumanMessage(content=item["content"]))
        else:
            history.append(AIMessage(content=item["content"]))
    return history

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
    apply_dashboard_theme()
    init_state()
    render_sidebar()

    # --- Header -----------------------------------------------------------
    st.markdown(
        """
        <section class="rag-hero">
            <div class="rag-eyebrow">LangChain · Retrieval augmented generation</div>
            <h1 class="rag-title">Chat with your <span>knowledge</span></h1>
            <p class="rag-subtitle">
                Upload your documents, build a semantic index, and get grounded answers
                from an agent that searches before it speaks.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    # --- Upload & process box ----------------------------------------------
    with st.container(border=True):
        st.markdown(
            '<div class="upload-label">Upload &amp; process one or more files</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    # --- Process button ----------------------------------------------------
    process_col, _spacer = st.columns([3, 7])
    with process_col:
        process_clicked = st.button(
            "Process documents",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

    if process_clicked:
        try:
            current_file_set = file_set_id(uploaded_files)
            with st.spinner("Reading, splitting, and indexing documents..."):
                documents = [d for item in uploaded_files for d in uploaded_file_to_documents(item)]
                agent, chunk_count = build_agent(documents)
            st.session_state.agent = agent
            st.session_state.file_set_id = current_file_set
            start_new_conversation()  # fresh chat for the new knowledge base
            st.success(f"Indexed {len(uploaded_files)} files ({len(documents)} pages) into {chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Could not process the documents: {exc}")

    if uploaded_files and "file_set_id" in st.session_state:
        if file_set_id(uploaded_files) != st.session_state.file_set_id:
            st.warning("The selected files changed. Process them before asking questions.")

    # --- Chat area ---------------------------------------------------------
    convo = get_active_conversation()
    for message in convo["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "agent" not in st.session_state:
        st.info("Upload and process at least one document to begin.")

    question = st.chat_input(
        "Ask questions about your documents",
        disabled="agent" not in st.session_state,
    )
    if question:
        history = to_langchain_history(convo["messages"])
        convo["messages"].append({"role": "user", "content": question})
        if convo["title"] == "New conversation":
            convo["title"] = question[:28] + ("…" if len(question) > 28 else "")
        db_save_message(convo, "user", question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer = stream_answer(
                st.session_state.agent, history + [HumanMessage(content=question)]
            )

        convo["messages"].append({"role": "assistant", "content": answer})
        db_save_message(convo, "assistant", answer)
        st.rerun()  

if __name__ == "__main__":
    main()
