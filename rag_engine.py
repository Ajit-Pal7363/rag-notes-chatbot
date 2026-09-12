"""
rag_engine.py
--------------
Core RAG (Retrieval-Augmented Generation) logic:
1. Load PDFs / text notes
2. Split into chunks
3. Create embeddings (local, free - sentence-transformers)
4. Retrieve most relevant chunks for a question
5. Send question + retrieved context to Claude to generate an answer
"""

import os
import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


class RAGEngine:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=50):
        """
        embedding_model_name: local embedding model (downloaded once, runs free & offline)
        chunk_size: approx number of characters per chunk
        chunk_overlap: overlap between chunks so context isn't cut off mid-thought
        """
        self.embedder = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks = []          # list of text chunks
        self.chunk_sources = []   # which file each chunk came from
        self.embeddings = None    # numpy array of chunk embeddings

        # Gemini client - reads GOOGLE_API_KEY from environment variable
        # Free tier: console at https://aistudio.google.com/apikey
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)

    # ---------- STEP 1: Load documents ----------

    def load_pdf(self, filepath):
        """Extract raw text from a PDF file."""
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text

    def load_txt(self, filepath):
        """Load raw text from a .txt or .md file."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def add_document(self, filepath):
        """Load a file (PDF or text), chunk it, and add to the knowledge base."""
        filename = os.path.basename(filepath)
        ext = filepath.lower().split(".")[-1]

        if ext == "pdf":
            text = self.load_pdf(filepath)
        elif ext in ("txt", "md"):
            text = self.load_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = self._chunk_text(text)
        self.chunks.extend(chunks)
        self.chunk_sources.extend([filename] * len(chunks))
        print(f"Loaded '{filename}': {len(chunks)} chunks added.")

    # ---------- STEP 2: Chunking ----------

    def _chunk_text(self, text):
        """
        Split text into overlapping chunks.
        Cleans whitespace first, then slides a window across the text.
        """
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    # ---------- STEP 3: Embeddings ----------

    def build_index(self):
        """Compute embeddings for all chunks. Call this after adding all documents."""
        if not self.chunks:
            raise ValueError("No documents loaded yet. Call add_document() first.")
        self.embeddings = self.embedder.encode(self.chunks, show_progress_bar=True, normalize_embeddings=True)
        print(f"Index built: {len(self.chunks)} chunks embedded.")

    # ---------- STEP 4: Retrieval ----------

    def retrieve(self, query, top_k=4):
        """Find the top_k most relevant chunks for a given query using cosine similarity."""
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        # embeddings are normalized, so dot product = cosine similarity
        scores = self.embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx],
                "source": self.chunk_sources[idx],
                "score": float(scores[idx])
            })
        return results

    # ---------- STEP 5: Generation ----------

    def answer(self, query, top_k=4, model="gemini-3.6-flash"):
        """
        Full RAG pipeline: retrieve relevant chunks, then ask Gemini to answer
        using ONLY that retrieved context.
        """
        retrieved = self.retrieve(query, top_k=top_k)

        context_block = "\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
        )

        system_prompt = (
            "You are a helpful assistant that answers questions using ONLY the "
            "provided context from the user's notes/documents. "
            "If the answer isn't in the context, say you don't know based on the "
            "provided documents - do not make up information. "
            "Cite which source file each part of your answer comes from."
        )

        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        gen_model = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
        response = gen_model.generate_content(user_message)

        answer_text = response.text
        return answer_text, retrieved


# ---------- Simple command-line test ----------
if __name__ == "__main__":
    engine = RAGEngine()

    # Example usage - replace with your own file paths
    # engine.add_document("my_notes.pdf")
    # engine.build_index()
    # ans, sources = engine.answer("What is the main topic of chapter 2?")
    # print(ans)

    print("RAG engine ready. Import this module or run app.py for the web UI.")
