import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import create_retriever_tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Charger .env depuis la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "Tu es un assistant Agentic RAG pédagogique en français. "
    "Utilise l'outil `recherche_base_connaissances` dès que la question porte sur le cours "
    "(LangGraph, RAG agentique, MCP ou EMSI)."
)


def _charger_documents() -> list[Document]:
    docs_dir = Path(__file__).resolve().parent / "docs"
    docs: list[Document] = []
    if docs_dir.exists():
        for txt_file in sorted(docs_dir.glob("*.txt")):
            docs.append(Document(page_content=txt_file.read_text(encoding="utf-8"), metadata={"source": txt_file.name}))
    if not docs:
        docs = [
            Document(
                page_content=(
                    "LangGraph permet de modéliser des workflows d'agents sous forme de graphes d'états "
                    "avec des nœuds, des arêtes conditionnelles, des boucles et de la persistance."
                ),
                metadata={"source": "cours_langgraph.txt"},
            ),
            Document(
                page_content=(
                    "Dans un RAG Agentique avec LangGraph, le nœud agent décide s'il doit appeler "
                    "l'outil retriever_tool (nœud tools) ou répondre directement à l'utilisateur."
                ),
                metadata={"source": "cours_rag_agentique.txt"},
            ),
            Document(
                page_content=(
                    "LangGraph Studio permet de visualiser et tester un graphe compilé en local "
                    "grâce au fichier langgraph.json et à la commande `langgraph dev`."
                ),
                metadata={"source": "cours_studio.txt"},
            ),
        ]
    return docs


def _creer_retriever_tool():
    docs = _charger_documents()
    embeddings = FastEmbedEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return create_retriever_tool(
        retriever=retriever,
        name="recherche_base_connaissances",
        description="Recherche des passages pertinents dans la base documentaire du cours.",
    )


def build_graph(checkpointer=None):
    retriever_tool = _creer_retriever_tool()
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0,
    ).bind_tools([retriever_tool])

    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode([retriever_tool]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


# Graphe exporté pour LangGraph Studio (langgraph.json)
graph = build_graph()


if __name__ == "__main__":
    question = "Comment fonctionne un RAG Agentique avec LangGraph ?"
    res = graph.invoke({"messages": [("user", question)]})
    print("Question :", question)
    print("Réponse  :", res["messages"][-1].content)
