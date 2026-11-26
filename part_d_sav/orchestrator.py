import asyncio
import json
import os
from pathlib import Path
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

GARANTIES_PATH = Path(__file__).resolve().parent / "data" / "garanties.json"

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)


@tool
def consulter_garantie(numero_serie: str) -> str:
    """Interroge en lecture seule le registre des garanties à partir d'un numéro de série."""
    registre = json.loads(GARANTIES_PATH.read_text(encoding="utf-8"))
    info = registre.get(numero_serie.strip().upper())
    if not info:
        return f"Numéro de série '{numero_serie}' introuvable dans le registre."
    return (
        f"Série : {numero_serie.upper()} | Modèle : {info['modele']} | "
        f"Achat : {info['date_achat']} | Fin garantie : {info['fin_garantie']} | Statut : {info['statut']}"
    )


class SAVState(TypedDict):
    demande: str
    image_ref: str
    categorie: str
    agent_utilise: str
    reponse_finale: str


def noeud_orchestrateur(state: SAVState) -> dict:
    # Si une réponse finale est déjà produite, l'orchestrateur ne re-route pas
    if state.get("reponse_finale"):
        return {}

    prompt = (
        "Classe cette demande SAV en exactement un seul mot : "
        "'documentation' (dépannage, voyant, pictogramme, manuel) "
        "ou 'garantie' (vérification de garantie, numéro de série).\n\n"
        f"Demande : {state['demande']} (image: {state.get('image_ref') or 'aucune'})"
    )
    cat = llm.invoke(prompt).content.strip().lower()
    categorie = "garantie" if "garantie" in cat else "documentation"
    return {"categorie": categorie}


def route_orchestrateur(state: SAVState) -> Literal["agent_documentaire", "agent_garantie", "__end__"]:
    if state.get("reponse_finale"):
        return "__end__"
    if state["categorie"] == "garantie":
        return "agent_garantie"
    return "agent_documentaire"


async def noeud_agent_documentaire(state: SAVState) -> dict:
    client = MultiServerMCPClient(
        {
            "sav_doc": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8001/mcp",
            }
        }
    )
    tools = await client.get_tools()
    rag_tool = next(t for t in tools if t.name == "rag_multimodal")
    reponse = await rag_tool.ainvoke({"question": state["demande"], "image_ref": state.get("image_ref", "")})
    if isinstance(reponse, list):
        reponse = "\n".join(b.get("text", "") for b in reponse if isinstance(b, dict))
    return {"agent_utilise": "agent_documentaire", "reponse_finale": str(reponse)}


def noeud_agent_garantie(state: SAVState) -> dict:
    llm_garantie = llm.bind_tools([consulter_garantie])
    msg = llm_garantie.invoke(
        f"Utilise consulter_garantie pour répondre à la demande : {state['demande']}"
    )
    if msg.tool_calls:
        outil_res = consulter_garantie.invoke(msg.tool_calls[0]["args"])
        reponse = llm.invoke(
            f"Formule une réponse client claire à partir de ce résultat : {outil_res}"
        ).content
    else:
        reponse = msg.content
    return {"agent_utilise": "agent_garantie", "reponse_finale": reponse}


def build_sav_graph():
    builder = StateGraph(SAVState)
    builder.add_node("orchestrateur", noeud_orchestrateur)
    builder.add_node("agent_documentaire", noeud_agent_documentaire)
    builder.add_node("agent_garantie", noeud_agent_garantie)

    builder.add_edge(START, "orchestrateur")
    builder.add_conditional_edges("orchestrateur", route_orchestrateur)
    builder.add_edge("agent_documentaire", "orchestrateur")
    builder.add_edge("agent_garantie", "orchestrateur")

    return builder.compile()


async def main():
    app = build_sav_graph()

    exemples = [
        {
            "demande": "Mon imprimante PX-200 affiche un voyant clignotant. Que signifie le pictogramme sur la photo jointe ?",
            "image_ref": "img_px200_01",
        },
        {
            "demande": "Est-ce que mon imprimante est encore sous garantie ? Son numéro de série est PX200-0042.",
            "image_ref": "",
        },
    ]

    for i, ex in enumerate(exemples, 1):
        res = await app.ainvoke({
            "demande": ex["demande"],
            "image_ref": ex["image_ref"],
            "categorie": "",
            "agent_utilise": "",
            "reponse_finale": "",
        })
        print(f"=== Exemple {i} ===")
        print(f"Catégorie     : {res['categorie']}")
        print(f"Agent utilisé : {res['agent_utilise']}")
        print(f"Réponse       :\n{res['reponse_finale']}\n")


if __name__ == "__main__":
    asyncio.run(main())
