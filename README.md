# 🎬 MovieMate AI — RAG Movie Chatbot

A minimal, responsive movie assistant that answers questions and gives smart recommendations from a curated IMDB‑style CSV. Built with Python, LangChain, Chroma, sentence‑transformers, Gemini, and Streamlit. It persists embeddings for fast reloads and provides a clean chat UI with context‑grounded answers.

> 🎯 Built for students and hobbyists to learn Retrieval‑Augmented Generation (RAG) end‑to‑end.

---

## ❓ Why This Project
Manually searching across multiple sites for movies, plots, and similar titles is slow and distracting. Copy‑pasting into a general chatbot often returns ungrounded or hallucinated answers.

MovieMate AI solves this by:
- Indexing a structured IMDB‑like dataset,
- Retrieving relevant facts per query,
- Grounding the LLM response strictly in context.

It’s accurate, fast, and easy to deploy.

---

## 🚀 Features
- ✅ RAG over an IMDB‑style CSV (title, year, genre, cast, overview)
- 🔎 Semantic retrieval with all‑MiniLM‑L6‑v2 embeddings
- 🧠 Context‑grounded answers using Gemini
- 💬 Streamlit chat UI with history
- ⚙️ Adjustable Top‑K retrieval from the sidebar
- 💾 Persistent vector index (Chroma) for quick restarts

---

<!-- ## 🖼️ Preview
- Ask: “Recommend a sci‑fi under 2 hours”
- Ask: “Movies similar to Andhadhun”
- Ask: “Best horror movies since 2015”

The app retrieves relevant rows from your CSV and produces concise, helpful responses.*/ -->

---

## 🛠️ Tech Stack
- Python 3.10+
- Streamlit (UI)
- LangChain (chains, retrieval)
- Chroma (vector DB)
- sentence‑transformers/all‑MiniLM‑L6‑v2 (embeddings)
- Gemini (via langchain‑google‑genai)

---

<!-- ## 🌐 Live Demo (Optional)
- Deploy on Streamlit Community Cloud or run locally (instructions below). -->

---

<!-- ## 📦 Project Structure -->

## 👨‍💻 Author
**Sarthak Maheshwari**

---

## 📚 Citation (Dataset)

```bibtex
@misc{bhagtani2023imdb,
  author       = {Deven Bhagtani},
  title        = {IMDB Movie Dataset (1951--2023)},
  year         = {2023},
  howpublished = {\url{https://github.com/devensinghbhagtani/Bollywood-Movie-Dataset}},
  note         = {GitHub repository}
}
```



---

## 📜 License
This project is for educational/demo use. Review and comply with the licenses/terms of the dataset, models, and any external APIs used.

© 2025 Sarthak Maheshwari

