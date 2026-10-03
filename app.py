"""Streamlit chatbot that answers questions from your own notes using an LLM (RAG)."""
import os

import anthropic
import streamlit as st

from rag import Index, load_chunks

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
SYSTEM = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite passages like [1] or [2]. If the context does not contain the answer, "
    "say you don't know instead of guessing."
)

st.set_page_config(page_title="Notes Chatbot", page_icon="💬")
st.title("Chat with your notes")

folder = st.sidebar.text_input("Notes folder", "notes")
top_k = st.sidebar.slider("Passages to retrieve", 1, 8, 4)


@st.cache_resource
def get_index(path):
    chunks = load_chunks(path)
    return Index(chunks) if chunks else None


if st.sidebar.button("Reload notes"):
    st.cache_resource.clear()

index = get_index(folder)
if index is None:
    st.warning(f"No .txt or .md files found in '{folder}'.")
    st.stop()
st.sidebar.caption(f"{len(index.chunks)} chunks indexed")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask something about your notes")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = index.search(question, k=top_k)
    context = "\n\n".join(f"[{i}] ({c.source})\n{c.text}" for i, (c, _) in enumerate(hits, 1))

    # Past turns are sent as plain text; only the newest question carries retrieved context.
    messages = st.session_state.messages[:-1] + [
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    def stream():
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        with client.messages.stream(
            model=MODEL, max_tokens=800, system=SYSTEM, messages=messages
        ) as s:
            yield from s.text_stream

    with st.chat_message("assistant"):
        answer = st.write_stream(stream())
        with st.expander("Sources"):
            for i, (c, score) in enumerate(hits, 1):
                st.markdown(f"**[{i}] {c.source}** (score {score:.2f})")
                st.caption(c.text[:300])
    st.session_state.messages.append({"role": "assistant", "content": answer})
