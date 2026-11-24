# RAGBot

Chat with any PDF using Google Gemini, LangChain, and Chroma, served through Streamlit.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Get a Gemini API key from https://aistudio.google.com/apikey, then either:

```bash
set GOOGLE_API_KEY=your-key      # Windows  (macOS/Linux: export GOOGLE_API_KEY=your-key)
```

or paste it into the sidebar when the app starts.

## Run

```bash
streamlit run app.py
```

Upload a PDF in the sidebar and ask questions. Answers come only from the document, and each one lists the pages it drew from.

## How it works

1. The PDF is split into 1000-character chunks with 200 characters of overlap.
2. The chunks are embedded with `gemini-embedding-001` and stored in an in-memory Chroma index.
3. For each question, the 4 most similar chunks are retrieved and passed to `gemini-2.5-flash` as context.
