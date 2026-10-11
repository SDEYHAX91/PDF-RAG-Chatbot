from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MODEL = "qwen2.5:7b"
NUM_CTX = 8192
TOP_K = 6               # chunks for a specific question
BROAD_SAMPLE = 10       # evenly spaced chunks for "summarize / main idea" questions
HISTORY_TURNS = 6       # last N chat messages passed to the model

NOT_FOUND = "Not found in the uploaded documents."

BROAD_RE = re.compile(
    r"\b(summar\w*|overview|main (idea|topic|point)s?|key (points?|topics?|takeaways?)|"
    r"outline|table of contents|topics covered|"
    r"what('s| is) (this|the) (document|pdf|file|book|text)( about)?|"
    r"(about|explain) (this|the) (document|pdf|file|book))\b",
    re.I,
)

SYSTEM_PROMPT = f"""You answer questions about the user's uploaded documents using ONLY the excerpts provided in the message.

Rules:
- Use only facts stated in the excerpts. Never use outside knowledge, even if you know the answer.
- If the excerpts do not contain the answer, reply with exactly: {NOT_FOUND}
- Never invent facts, page numbers, file names or quotes. Source names are added automatically, so do not write them.
- Use as many bullets as the excerpts support (1-6). Do not pad to reach a number.

Format:
1. A short direct answer (2-3 sentences) in plain language.
2. One lead-in line ending with a colon, for example "Key points:" or "How it works:".
3. A bulleted list using "* ". Each bullet starts with a short bold label and a colon, then one or two sentences of specifics from the excerpts.
Do not add a closing summary, extra headings, or the words "Query" or "Answer".
Use LaTeX with $$...$$ for math if needed."""


# ----------------------------------------------------------------------------- embeddings
class NomicEmbeddings(OllamaEmbeddings):
    """nomic-embed-text is trained with task prefixes; without them retrieval is noticeably worse."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")


# ----------------------------------------------------------------------------- PDF reading
def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def strip_repeated_headers(pages: list[str], top_lines: int = 3, min_share: float = 0.4) -> list[str]:
    """Remove running headers/footers (e.g. 'Book Title' / 'Page 8') that repeat on most pages.

    They carry no content but get embedded into every chunk, which makes all chunks look alike
    and flattens the similarity scores.
    """
    if len(pages) < 3:
        return pages
    counts: Counter[str] = Counter()
    for text in pages:
        lines = [l for l in text.splitlines() if l.strip()]
        counts.update({_norm(l) for l in lines[:top_lines]})
        counts.update({_norm(l) for l in lines[-2:]})
    repeated = {k for k, v in counts.items() if v / len(pages) >= min_share}

    cleaned = []
    for text in pages:
        lines = text.splitlines()
        nonblank = [i for i, l in enumerate(lines) if l.strip()]
        drop = {i for i in nonblank[:top_lines] + nonblank[-2:] if _norm(lines[i]) in repeated}
        cleaned.append("\n".join(l for i, l in enumerate(lines) if i not in drop))
    return cleaned


def file_to_documents(name: str, raw: bytes) -> list[Document]:
    """One Document per PDF page (so chunks keep a page number); one Document for text files."""
    suffix = name.rsplit(".", 1)[-1].lower()
    if suffix == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        pages = strip_repeated_headers([p.extract_text() or "" for p in reader.pages])
        docs = [
            Document(page_content=t, metadata={"source": name, "page": i + 1})
            for i, t in enumerate(pages)
            if t.strip()
        ]
    else:
        text = raw.decode("utf-8", errors="replace")
        docs = [Document(page_content=text, metadata={"source": name})] if text.strip() else []
    if not docs:
        raise ValueError(f"No readable text was found in {name}.")
    return docs


# ----------------------------------------------------------------------------- pipeline
class RagPipeline:
    def __init__(self, vector_store: InMemoryVectorStore, chunks: list[Document], llm):
        self.vector_store = vector_store
        self.chunks = chunks
        self.llm = llm

    # --- retrieval ---------------------------------------------------------
    def retrieve(self, question: str, history: list | None = None) -> tuple[list[Document], bool]:
        """Return (chunks, is_broad). Broad questions get an even sample across the documents,
        because similarity search cannot rank chunks for 'what is this document about'."""
        if BROAD_RE.search(question):
            n = min(BROAD_SAMPLE, len(self.chunks))
            step = len(self.chunks) / n
            return [self.chunks[int(i * step)] for i in range(n)], True

        query = question
        if history and len(question.split()) < 8:  # short follow-up like "explain more"
            last_user = next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), "")
            query = f"{last_user} {question}".strip()
        hits = self.vector_store.similarity_search_with_score(query, k=min(TOP_K, len(self.chunks)))
        return [d for d, _ in hits], False

    # --- generation --------------------------------------------------------
    def build_messages(self, question: str, history: list, docs: list[Document], broad: bool) -> list:
        parts = []
        for i, d in enumerate(docs, 1):
            page = d.metadata.get("page")
            where = f"{d.metadata['source']}" + (f", page {page}" if page else "")
            parts.append(f"[Excerpt {i} - {where}]\n{d.page_content}")
        note = (
            "The excerpts below are sampled evenly across the whole document. Give an overview.\n\n"
            if broad else ""
        )
        user = f"{note}Excerpts:\n\n" + "\n\n---\n\n".join(parts) + f"\n\nQuestion: {question}"
        return [SystemMessage(content=SYSTEM_PROMPT), *history[-HISTORY_TURNS:], HumanMessage(content=user)]

    def stream(self, question: str, history: list, docs: list[Document], broad: bool) -> Iterator[str]:
        for chunk in self.llm.stream(self.build_messages(question, history, docs, broad)):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                yield text

    # --- citations written by code, not by the model -----------------------
    @staticmethod
    def sources_line(docs: list[Document]) -> str:
        pages: dict[str, set] = defaultdict(set)
        for d in docs:
            pages[d.metadata.get("source", "unknown")]
            if d.metadata.get("page"):
                pages[d.metadata["source"]].add(d.metadata["page"])
        bits = []
        for src, pgs in pages.items():
            bits.append(f"{src} (p. {', '.join(map(str, sorted(pgs)))})" if pgs else src)
        return "\n\n**Sources:** " + "; ".join(bits)


def build_pipeline(documents: list[Document], embeddings=None, llm=None) -> tuple[RagPipeline, int]:
    embeddings = embeddings or NomicEmbeddings(model="nomic-embed-text")
    llm = llm or ChatOllama(model=MODEL, temperature=0, num_ctx=NUM_CTX)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    return RagPipeline(store, chunks, llm), len(chunks)