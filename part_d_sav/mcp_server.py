import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

DATA_DIR = Path(__file__).resolve().parent / "data"
MANUEL_PATH = DATA_DIR / "manuel_px200.txt"

# Construction de l'index vectoriel sur les sections et figures du manuel
blocs = [b.strip() for b in MANUEL_PATH.read_text(encoding="utf-8").split("\n\n") if b.strip()]
docs = [Document(page_content=b, metadata={"source": f"manuel_px200.txt#bloc_{i+1}"}) for i, b in enumerate(blocs)]
vectorstore = FAISS.from_documents(docs, FastEmbedEmbeddings())

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)

mcp = FastMCP("SAVDocServer", host="127.0.0.1", port=8001)


@mcp.tool()
def rag_multimodal(question: str, image_ref: str = "") -> str:
    """Recherche des passages et figures dans le manuel technique et produit une réponse sourcée."""
    requete = f"{question} {image_ref}".strip()
    passages = vectorstore.similarity_search(requete, k=2)
    contexte = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in passages)

    prompt = (
        "Tu es un assistant SAV technique. Réponds de façon claire et sourcée "
        "(cite la section ou la figure) uniquement à partir du contexte ci-dessous.\n\n"
        f"Contexte :\n{contexte}\n\n"
        f"Question : {question}\nRéférence image : {image_ref or 'aucune'}"
    )
    return llm.invoke(prompt).content


if __name__ == "__main__":
    print("Serveur MCP SAV (rag_multimodal) démarré sur http://127.0.0.1:8001/mcp")
    mcp.run(transport="streamable-http")
