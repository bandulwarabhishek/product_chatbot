import streamlit as st
import uuid
from prometheus_client import CollectorRegistry, Counter, generate_latest

from product.data_ingestion import DataIngestion
from product.rag_agent import RAGAgentBuilder

from dotenv import load_dotenv
load_dotenv()



# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# ── Prometheus metrics ────────────────────────────────────────────────────────
@st.cache_resource
def create_metrics():
    registry = CollectorRegistry()
    request_count = Counter("http_requests_total", "Total HTTP Requests", registry=registry)
    prediction_count = Counter("model_predictions_total", "Total Model Predictions", registry=registry)
    return registry, request_count, prediction_count

METRICS_REGISTRY, REQUEST_COUNT, PREDICTION_COUNT = create_metrics()

# ── Session-scoped singletons (created once per browser tab) ──────────────────
@st.cache_resource
def load_vector_store():
    return DataIngestion().ingest(load_existing=True)

@st.cache_resource
def load_rag_agent(_vector_store):          # leading _ keeps Streamlit from hashing it
    return RAGAgentBuilder(_vector_store).build_agent()

vector_store = load_vector_store()
rag_agent    = load_rag_agent(vector_store)

# ── Per-session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    print(f"[INFO] New chat thread created: {st.session_state.thread_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []          # {"role": "user"|"assistant", "content": "..."}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask me anything about our products…"):
    REQUEST_COUNT.inc()

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call RAG agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = rag_agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            PREDICTION_COUNT.inc()

            answer = (
                response["messages"][-1].content
                if response.get("messages")
                else "Sorry, I couldn't find relevant product information."
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Sidebar: debug / metrics ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Session info")
    st.code(st.session_state.thread_id, language=None)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    with st.expander("📊 Prometheus metrics"):
        st.text(generate_latest().decode())