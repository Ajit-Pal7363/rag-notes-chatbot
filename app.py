"""
app.py
------
Streamlit UI for the RAG chatbot.

Run with:
    streamlit run app.py

Make sure GOOGLE_API_KEY is set as an environment variable before running:
    export GOOGLE_API_KEY="your-key-here"      (Mac/Linux)
    setx GOOGLE_API_KEY "your-key-here"         (Windows)

Get a free key at: https://aistudio.google.com/apikey
"""

import os
import tempfile
import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="Notes Q&A Bot", page_icon="📚", layout="wide")

st.title("📚 Chat with Your Notes / PDFs")
st.caption("Upload your PDFs or text notes, then ask questions. Answers come only from your documents.")

# ---------- Session state setup ----------
if "engine" not in st.session_state:
    st.session_state.engine = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Sidebar: API key + file upload ----------
with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "Google (Gemini) API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Free — get one at aistudio.google.com/apikey"
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Build Knowledge Base", type="primary", disabled=not uploaded_files):
        if not api_key:
            st.error("Please enter your Google (Gemini) API key first.")
        else:
            with st.spinner("Reading files and building index... (first run downloads the embedding model, ~90MB)"):
                engine = RAGEngine()
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix="_" + uploaded_file.name) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    engine.add_document(tmp_path)
                    os.unlink(tmp_path)

                engine.build_index()
                st.session_state.engine = engine
                st.session_state.indexed = True
            st.success(f"Knowledge base ready! {len(engine.chunks)} chunks indexed from {len(uploaded_files)} file(s).")

    if st.session_state.indexed:
        st.info(f"✅ {len(st.session_state.engine.chunks)} chunks loaded and ready.")

    st.divider()
    top_k = st.slider("Chunks to retrieve per question", 1, 10, 4)

# ---------- Main chat area ----------
if not st.session_state.indexed:
    st.info("👈 Upload your files and click 'Build Knowledge Base' in the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("View sources used"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['source']}** (relevance: {src['score']:.2f})")
                        st.text(src["text"][:300] + "...")

    # Chat input
    if question := st.chat_input("Ask a question about your notes..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = st.session_state.engine.answer(question, top_k=top_k)
                    st.markdown(answer)
                    with st.expander("View sources used"):
                        for src in sources:
                            st.markdown(f"**{src['source']}** (relevance: {src['score']:.2f})")
                            st.text(src["text"][:300] + "...")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
