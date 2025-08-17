import os
import time
import streamlit as st

from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---------------- Config ----------------
DATA_PATH = "IMDB-Movie-Dataset(2023-1951) (1).csv" 
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K = 6

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")

st.set_page_config(page_title="MovieMate AI", page_icon="🎬")
st.title("MovieMate AI 🎬")
st.caption("RAG-based movie Q&A and recommendations")

# ---------------- Build or load vector store ----------------
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # If persisted DB exists, load it
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        return vs

    # Else, build from CSV
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}. Please add it to the repo.")
        st.stop()

    loader = CSVLoader(file_path=DATA_PATH, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vs.persist()
    return vs

with st.spinner("Loading index..."):
    vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_K})

# ---------------- LLM and chains ----------------
if not GOOGLE_API_KEY:
    st.warning("Add GOOGLE_API_KEY in Streamlit secrets to enable Gemini.")
else:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

    prompt = ChatPromptTemplate.from_template("""
You are MovieMate AI, a helpful movie assistant.
Use ONLY the context to answer or recommend. If unsure, say you don’t know.
If recommending, list 3–5 titles with 1–2 lines each, concise and friendly.

Context:
{context}

Question: {input}

Answer:
""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    def answer_query(q: str) -> str:
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        out = retrieval_chain.invoke({"input": q})
        return out.get("answer", "").strip()

    # ---------------- Sidebar filter ----------------
    st.sidebar.header("Search settings")
    k = st.sidebar.slider("Top-K retrieved chunks", 2, 12, DEFAULT_K)

    # Update retriever k live
    retriever.search_kwargs["k"] = k

    st.sidebar.markdown("Examples:")
    st.sidebar.code("Recommend a sci‑fi under 2 hours\nMovies similar to Inception\nBest horror movie since 2015")

    # ---------------- Chat UI ----------------
    if "history" not in st.session_state:
        st.session_state.history = []

    user_query = st.chat_input("Ask about movies or request recommendations…")
    # Display previous conversation
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    if user_query:
        st.session_state.history.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                with st.spinner("Thinking..."):
                    resp = answer_query(user_query)
                if not resp:
                    resp = "I couldn’t find enough info in the dataset. Try rephrasing or asking something else."
                placeholder.markdown(resp)
                st.session_state.history.append(("assistant", resp))
            except Exception as e:
                placeholder.error(f"Error: {e}")
