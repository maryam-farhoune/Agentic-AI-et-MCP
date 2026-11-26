import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from graph import build_graph

st.set_page_config(page_title="Chatbot Agentic RAG — LangGraph", page_icon="🤖")
st.title("Chatbot Agentic RAG (LangGraph)")

if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = build_graph(checkpointer=InMemorySaver())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez une question sur LangGraph ou le RAG agentique..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            result = st.session_state.rag_graph.invoke(
                {"messages": [("user", prompt)]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            answer = result["messages"][-1].content
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
