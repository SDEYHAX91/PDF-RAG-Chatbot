A private, fully local "chat with your documents" app. Upload PDFs or text files, ask questions, and get answers based **only** on those files, with source file names and page numbers shown.

Built with Streamlit, LangChain and Ollama. No API keys and no data leaves your machine.

<img width="1920" height="992" alt="image" src="https://github.com/user-attachments/assets/b6a9b0eb-144f-432b-9998-7e4f57454960" />

## Features

- Upload one or more files (`.pdf`, `.txt`, `.md`)
- Answers streamed token by token in a fixed format: short answer, lead-in line, bold-label bullets
- Source citations (file + page numbers) added by code, not by the model, so they can't be invented
- Replies `Not found in the uploaded documents.` when the answer isn't in the files
- Handles broad questions ("summarize", "what is this document about") by sampling chunks evenly across the document
- Conversation history saved in SQLite, with a sidebar to switch, delete, or clear all chats

## How it works

1. **Ingest:** PDFs are read page by page (so every chunk keeps its page number), and repeated headers/footers are stripped.
2. **Chunk and embed:** Text is split into 1000-character chunks (200 overlap) and embedded with `nomic-embed-text` into an in-memory vector store.
3. **Retrieve:** The code always retrieves, so the model can't skip the search step. Specific questions get the top 6 similar chunks; broad questions get 10 evenly spaced chunks.
4. **Generate:** A local `qwen2.5:7b` model (via Ollama, temperature 0) answers using only the retrieved excerpts.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Streamlit UI, chat flow, SQLite conversation history |
| `rag_pipeline.py` | PDF reading, chunking, embeddings, retrieval, prompt, citations |
| `requirements.txt` | Python dependencies |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

Pull the models:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## Setup and run

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
streamlit run main.py
```

Open the URL Streamlit prints (usually http://localhost:8501), upload your files, click **Process documents**, and start asking questions.

## Configuration

Edit the constants at the top of `rag_pipeline.py`:

| Constant | Default | Meaning |
|---|---|---|
| `MODEL` | `qwen2.5:7b` | Ollama chat model |
| `NUM_CTX` | `8192` | Context window |
| `TOP_K` | `6` | Chunks retrieved for specific questions |
| `BROAD_SAMPLE` | `10` | Chunks sampled for summary-style questions |
| `HISTORY_TURNS` | `6` | Previous messages sent to the model |

## Notes

- The vector store is in memory, so documents must be re-processed after restarting the app.
- Scanned PDFs without a text layer can't be read (no OCR).
- Chat history is stored in `chat_history.db` next to `main.py`. It may contain content from your documents, so don't commit it.
