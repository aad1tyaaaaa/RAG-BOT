"""RAGBot: chat with a PDF using Gemini + Chroma."""

import hashlib
import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the user's question using only the context below. "
     "If the answer is not in the context, say you don't know. Keep the answer concise.\n\n"
     "Context:\n{context}"),
    ("human", "{input}"),
])

st.set_page_config(page_title="RAGBot", page_icon="📄")
st.title("📄 RAGBot")
st.caption("Upload a PDF and ask questions about it.")

with st.sidebar:
    api_key = os.getenv("GOOGLE_API_KEY") or st.text_input("Gemini API key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if not api_key:
    st.info("Set GOOGLE_API_KEY or enter your Gemini API key in the sidebar.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key


@st.cache_resource(show_spinner="Indexing document...")
def build_retriever(file_hash: str, data: bytes):
    # file_hash is the cache key, so a new upload gets a fresh index.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
    try:
        docs = PyPDFLoader(f.name).load()
    finally:
        os.remove(f.name)
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    if not splits:
        raise ValueError("No extractable text found in this PDF.")
    store = Chroma.from_documents(splits, GoogleGenerativeAIEmbeddings(model=EMBED_MODEL),
                                  collection_name=f"pdf_{file_hash[:16]}")
    return store.as_retriever(search_kwargs={"k": 4})


if not uploaded_file:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

data = uploaded_file.getvalue()
file_hash = hashlib.sha256(data).hexdigest()
try:
    retriever = build_retriever(file_hash, data)
except Exception as e:
    st.error(f"Could not index the PDF: {e}")
    st.stop()

# Reset chat when a different file is uploaded.
if st.session_state.get("file_hash") != file_hash:
    st.session_state.file_hash = file_hash
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if question := st.chat_input("Ask a question about your document"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        docs = []
        try:
            with st.spinner("Thinking..."):
                docs = retriever.invoke(question)
                context = "\n\n".join(d.page_content for d in docs)
                llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
                answer = llm.invoke(PROMPT.format_messages(context=context, input=question)).content
        except Exception as e:
            answer = f"⚠️ Error: {e}"
        st.markdown(answer)
        if docs:
            with st.expander("Sources"):
                for d in docs:
                    st.markdown(f"**Page {d.metadata.get('page', 0) + 1}:** {d.page_content[:300]}…")
    st.session_state.messages.append({"role": "assistant", "content": answer})
