import streamlit as st
import os
import tempfile
import logging
from typing import List

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 6
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Document RAG Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Premium Look ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #e0e0e0 !important;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}

.main-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.main-header p {
    color: #a0aec0;
    font-size: 1.1rem;
    font-weight: 300;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0.5rem 0;
}

.status-ready {
    background: rgba(72, 187, 120, 0.15);
    color: #48bb78;
    border: 1px solid rgba(72, 187, 120, 0.3);
}

.status-waiting {
    background: rgba(237, 137, 54, 0.15);
    color: #ed8936;
    border: 1px solid rgba(237, 137, 54, 0.3);
}

.source-box {
    background: rgba(255,255,255, 0.03);
    border: 1px solid rgba(255,255,255, 0.08);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #cbd5e0;
}

.stChatMessage {
    background: rgba(255,255,255, 0.02) !important;
    border: 1px solid rgba(255,255,255, 0.05) !important;
    border-radius: 12px !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(102, 126, 234, 0.4) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(102, 126, 234, 0.8) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.sidebar-info {
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 8px;
    padding: 0.8rem;
    font-size: 0.85rem;
    color: #a0aec0;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Lazy-load expensive models ─────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_groq_llm():
    from langchain_groq import ChatGroq
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        st.error("⚠️ Groq API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop()
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=4096,
    )


# ─── Document Processing (inline) ───────────────────────────────
def process_pdf(uploaded_file) -> List[Document]:
    """Save uploaded file to temp, extract text, chunk it."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            st.error("The PDF appears to be empty or unreadable.")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Extracted {len(documents)} pages → {len(chunks)} chunks")
        return chunks
    finally:
        os.remove(tmp_path)


def build_vector_store(chunks: List[Document]):
    """Create an in-memory FAISS vector store from chunks."""
    embeddings = get_embedding_model()
    return FAISS.from_documents(chunks, embeddings)


# ─── RAG Pipeline (inline) ──────────────────────────────────────
def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get('page', 'N/A')
        parts.append(f"--- Source Chunk {i+1} (Page {page_num}) ---\n{doc.page_content}")
    return "\n\n".join(parts)

RAG_SYSTEM_TEMPLATE = """You are a highly detailed AI legal assistant. You must answer questions using ONLY the provided context from legal documents.

INSTRUCTIONS:
1. Read ALL the context chunks carefully and thoroughly.
2. Provide a COMPLETE and DETAILED answer that includes ALL relevant information found in the context, UNLESS the user explicitly asks for a summary.
3. If the user asks for a summary, provide a clear, concise, and accurate summary of the context within the requested length.
4. If the context contains a list of items (like laws, rules, regulations, or points), you MUST include ALL items from the list, not just the first few (unless summarizing).
5. Format your answer clearly using numbered lists, bullet points, or paragraphs as appropriate.
6. If the context does not contain the answer at all, respond EXACTLY with: "Information not found in document"
7. Do NOT use any external knowledge. Only use what is explicitly stated in the context below.

Context from the document:
{context}"""


CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


def answer_question(question: str, vector_store, chat_history: list) -> dict:
    """Run Conversational RAG: retrieve → format → LLM → answer."""
    llm = get_groq_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 1. Create a history-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Create the document chain (the actual Q&A logic)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_TEMPLATE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Create the final retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    try:
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})
    except Exception as e:
        logger.error(f"LLM error: {e}")
        # Fallback to simple search if chain fails
        retrieved_docs = vector_store.similarity_search(question, k=RETRIEVER_K)
        return {"answer": f"Error generating answer: {e}", "sources": retrieved_docs}

    return {"answer": response["answer"].strip(), "sources": response["context"]}


# ─── UI ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚖️ Legal Document RAG Chatbot</h1>
    <p>Upload a legal PDF and ask questions — answers are strictly based on the document contents.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

    if st.button("🔍 Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                chunks = process_pdf(uploaded_file)
                if chunks:
                    vs = build_vector_store(chunks)
                    st.session_state.vector_store = vs
                    st.session_state.doc_name = uploaded_file.name
                    st.success(f"✅ Indexed **{len(chunks)}** chunks from `{uploaded_file.name}`")
        else:
            st.warning("Please upload a PDF file first.")

    st.markdown("---")

    # Status indicator
    if "vector_store" in st.session_state:
        st.markdown(
            f'<div class="status-badge status-ready">🟢 Ready — {st.session_state.get("doc_name", "Document")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-waiting">🟡 Waiting for document</div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="sidebar-info">
        <strong>How it works:</strong><br>
        1. Upload a legal PDF<br>
        2. Click "Process Document"<br>
        3. Ask questions in the chat<br><br>
        <em>Powered by Groq LLM &amp; FAISS</em>
    </div>
    """, unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📑 View Sources"):
                for idx, src in enumerate(msg["sources"]):
                    page = src.get("page", "N/A")
                    st.markdown(f'<div class="source-box"><strong>Source {idx+1} (Page {page})</strong><br>{src["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "vector_store" not in st.session_state:
        err = "⚠️ Please upload and process a PDF document first."
        st.chat_message("assistant").warning(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
    else:
        with st.chat_message("assistant"):
                # Format chat history for LangChain
                from langchain_core.messages import HumanMessage, AIMessage
                formatted_history = []
                # Only pass the last 10 messages (5 turns) to save tokens
                for msg in st.session_state.messages[-10:]:
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_history.append(AIMessage(content=msg["content"]))

                result = answer_question(prompt, st.session_state.vector_store, formatted_history[:-1]) # exclude the current prompt
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                serialized_sources = []
                if sources:
                    with st.expander("📑 View Sources"):
                        for idx, doc in enumerate(sources):
                            page = doc.metadata.get("page", "N/A")
                            content = doc.page_content
                            st.markdown(f'<div class="source-box"><strong>Source {idx+1} (Page {page})</strong><br>{content}</div>', unsafe_allow_html=True)
                            serialized_sources.append({"content": content, "page": page})

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": serialized_sources,
                })
