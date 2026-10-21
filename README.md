# 📚 Notes/PDF Q&A Chatbot (RAG Project)

Upload your PDFs or notes and ask questions about them. The bot answers only from your documents, which reduces hallucination.

## How It Works (RAG Pipeline)

1. **Load**: Extracts raw text from PDF/text files
2. **Chunk**: Splits long text into smaller overlapping pieces
3. **Embed**: Converts each chunk into a number vector (local model, free)
4. **Retrieve**: When a question comes in, finds the most relevant chunks (cosine similarity)
5. **Generate**: Sends the relevant chunks and the question to Gemini, which answers based on those chunks

## Setup (Up and Running in 5 Minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a Google Gemini API key (FREE)
- Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and log in with your Google account
- Click "Create API Key". It is free, and no card is required
- The free tier has rate limits (they change over time, so check Google AI Studio for the current numbers), which is enough for a personal project

### 3. Run the app
```bash
streamlit run app.py
```

The app opens in your browser (usually at `http://localhost:8501`).

### 4. Use it
1. Enter your API key in the sidebar
2. Upload PDF/txt files
3. Click "Build Knowledge Base" (the first run takes a bit longer because the embedding model, about 90 MB, is downloaded once)
4. Start asking questions in the chat box below

## Files Explained

| File | Purpose |
|---|---|
| `rag_engine.py` | Core logic: PDF reading, chunking, embeddings, retrieval, answer generation |
| `app.py` | Streamlit web UI |
| `requirements.txt` | Required Python packages |

## Using It from the Command Line

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-key-here"

from rag_engine import RAGEngine

engine = RAGEngine()
engine.add_document("my_notes.pdf")
engine.add_document("chapter2.pdf")
engine.build_index()

answer, sources = engine.answer("What is the main topic of Chapter 2?")
print(answer)
```

## Ideas to Extend This Project (Make It Portfolio-Ready)

- **Better embeddings**: Try `all-mpnet-base-v2` instead of `all-MiniLM-L6-v2` (better quality, slightly slower)
- **Vector DB**: For larger documents, use **ChromaDB** or **FAISS** instead of numpy (production-grade)
- **Multi-turn memory**: Pass chat history as context so follow-up questions are understood
- **Citations UI**: Show the exact page number along with each answer
- **Deploy**: Deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting your GitHub repo

## Common Issues

- **"No module named 'X'"**: Run `pip install -r requirements.txt` again
- **Slow first run**: The embedding model is being downloaded, which only happens once
- **API key error**: Check that you entered the correct key in the sidebar, or set the `GOOGLE_API_KEY` environment variable
- **Rate limit error**: The Gemini free tier has a requests-per-minute limit, so wait a bit and try again

---
Built with Python, Streamlit and the Gemini API.
