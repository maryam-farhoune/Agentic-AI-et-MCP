import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
)


async def main():
    client = MultiServerMCPClient(
        {
            "demo": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:8000/mcp",
            }
        }
    )
    tools = await client.get_tools()
    print("Outils MCP chargés :", [t.name for t in tools])

    agent = create_agent(model=llm, tools=tools)
    question = "Consulte les infos de Mohamed puis calcule son salaire net."
    result = await agent.ainvoke({"messages": [("user", question)]})
    print("\nQuestion :", question)
    print("Réponse  :", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
