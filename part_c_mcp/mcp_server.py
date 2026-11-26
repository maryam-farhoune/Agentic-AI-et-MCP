from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoMCPServer", host="127.0.0.1", port=8000)

EMPLOYES = {
    "mohamed": {"poste": "Ingénieur IA", "departement": "R&D", "salaire_brut": 18000},
    "sara": {"poste": "Data Scientist", "departement": "Analytics", "salaire_brut": 16500},
    "youssef": {"poste": "Architecte Cloud", "departement": "Infrastructure", "salaire_brut": 20000},
}


@mcp.tool()
def consulter_employe(nom: str) -> str:
    """Retourne les informations d'un employé (poste, département, salaire brut) à partir de son prénom."""
    info = EMPLOYES.get(nom.strip().lower())
    if not info:
        return f"Aucun employé trouvé pour '{nom}'."
    return (
        f"Employé : {nom.title()} | Poste : {info['poste']} | "
        f"Département : {info['departement']} | Salaire brut : {info['salaire_brut']} MAD"
    )


@mcp.tool()
def calculer_salaire_net(salaire_brut: float, taux_charges: float = 0.22) -> str:
    """Calcule le salaire net mensuel à partir du salaire brut et du taux de charges."""
    net = salaire_brut * (1 - taux_charges)
    return f"Salaire net estimé : {net:.2f} MAD (brut={salaire_brut:.2f}, charges={taux_charges:.0%})"


if __name__ == "__main__":
    print("Serveur MCP démarré sur http://127.0.0.1:8000/mcp (streamable-http)")
    mcp.run(transport="streamable-http")
