"""Web UI for the ROS 2 docs assistant.  Run:  streamlit run app.py"""
import streamlit as st

from src.rag import RAG

st.set_page_config(page_title="ROS 2 Docs Assistant", page_icon="🤖")
st.title("🤖 ROS 2 Docs Assistant")
st.caption("Ask questions about the ROS 2 (Humble) documentation. Answers come only from the docs.")


@st.cache_resource(show_spinner="Loading models and index...")
def load_rag():
    return RAG()          # loads embedding model + FAISS index once


rag = load_rag()


def show_sources(hits):
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)})"):
        for n, h in enumerate(hits, 1):
            st.markdown(f"**[{n}] {h['title']} > {h['section']}**  \n"
                        f"similarity: `{h['score']:.2f}` · [open page]({h['url']})")
            st.code(h["text"][:600], language=None)


with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (top-k)", 1, 10, 5)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("**Try asking:**")
    st.markdown("- How do I write a publisher node in Python?\n"
                "- Difference between topics and services?\n"
                "- How do I create a launch file?")

if "messages" not in st.session_state:
    st.session_state.messages = []

# redraw previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("hits"))

if question := st.chat_input("Ask about ROS 2..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs and thinking..."):
            try:
                answer, hits = rag.ask(question, k)
            except Exception as e:
                answer, hits = f"Something went wrong: `{e}`", []
        st.markdown(answer)
        show_sources(hits)

    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
