# 📚 Notes/PDF Q&A Chatbot (RAG Project)

Apne PDFs ya notes upload karo, aur unse related questions poocho — bot sirf tumhare documents se answer dega (hallucination kam hoga).

## Ye Kaise Kaam Karta Hai (RAG Pipeline)

1. **Load** — PDF/text file se raw text nikalta hai
2. **Chunk** — Lambe text ko chhote overlapping pieces mein todta hai
3. **Embed** — Har chunk ko ek number-vector mein convert karta hai (local model, free)
4. **Retrieve** — Jab question aata hai, sabse relevant chunks dhoondta hai (cosine similarity)
5. **Generate** — Relevant chunks + question ko Claude ko bhejta hai, jo un chunks ke base pe answer deta hai

## Setup (5 minute mein chalu)

### 1. Dependencies install karo
```bash
pip install -r requirements.txt
```

### 2. Google Gemini API key lo (FREE)
- [aistudio.google.com/apikey](https://aistudio.google.com/apikey) pe jaake Google account se login karo
- "Create API Key" pe click karo — bilkul free hai, koi card nahi chahiye
- Free tier limits: ~15 requests/minute, 1500 requests/din — personal project ke liye kaafi hai

### 3. App chalao
```bash
streamlit run app.py
```

Browser mein app khul jayega (usually `http://localhost:8501`).

### 4. Use karo
1. Sidebar mein apni API key daalo
2. PDF/txt files upload karo
3. "Build Knowledge Base" pe click karo (pehli baar thoda time lagega — embedding model download hoga, ~90MB, ek hi baar hota hai)
4. Neeche chat box mein questions poochna shuru karo!

## Files Explained

| File | Kaam |
|---|---|
| `rag_engine.py` | Core logic — PDF reading, chunking, embeddings, retrieval, answer generation |
| `app.py` | Streamlit web UI |
| `requirements.txt` | Zaroori Python packages |

## Command Line Se Bhi Use Kar Sakte Ho

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-key-here"

from rag_engine import RAGEngine

engine = RAGEngine()
engine.add_document("my_notes.pdf")
engine.add_document("chapter2.pdf")
engine.build_index()

answer, sources = engine.answer("Chapter 2 ka main topic kya hai?")
print(answer)
```

## Customization Ideas (Portfolio ke liye impressive banane ke liye)

- **Better embeddings**: `all-MiniLM-L6-v2` ki jagah `all-mpnet-base-v2` try karo (better quality, thoda slow)
- **Vector DB**: Bade documents ke liye numpy ki jagah **ChromaDB** ya **FAISS** use karo (production-grade)
- **Multi-turn memory**: Chat history ko bhi context mein pass karo taaki follow-up questions samajh sake
- **Citations UI**: Answer ke saath exact page number bhi dikhao
- **Deploy**: [Streamlit Community Cloud](https://streamlit.io/cloud) pe free deploy kar sakte ho — GitHub repo connect karke

## Common Issues

- **"No module named 'X'"** → `pip install -r requirements.txt` phir se chalao
- **Slow first run** → Embedding model download ho raha hai, ek hi baar hota hai
- **API key error** → Sidebar mein sahi key daali hai check karo, ya `GOOGLE_API_KEY` environment variable set karo
- **Rate limit error** → Gemini free tier mein ~15 requests/minute ki limit hai, thoda ruk ke phir try karo

---
Made with Claude 🤖
