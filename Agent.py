"""
CreditRisk Copilot : agent d'audit de données de crédit

L'utilisateur donne un fichier CSV et pose des questions.
L'agent choisit lui-même quels outilst de qualité de données (data_quality_tools.py) lancer,
puis explique les résultats.

Confidentialité : le LLM ne voit JAMAIS les lignes du fichier. Il reçoit
seulement les noms des colonnes et les résultats agrégés des outils.

"""

import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError


import data_quality_tools as DQT

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_TOURS = 8  # garde-fou : nombre maximum d'allers-retours avec le LLM par question

#################################################################################
###################Chargement du fichier (les données restent sur CETTE machine)

CHEMIN_CSV = sys.argv[1] if len(sys.argv) > 1 else "credit_card_default.csv"
COLONNE_DEFAUT = sys.argv[2] if len(sys.argv) > 2 else "default_payment_next_month"

df = pd.read_csv(CHEMIN_CSV)
if COLONNE_DEFAUT not in df.columns:
    sys.exit(
        f"La colonne de défaut '{COLONNE_DEFAUT}' n'existe pas dans {CHEMIN_CSV}.\n"
        f"Colonnes disponibles : {list(df.columns)}\n"
        f"Relance en précisant la bonne : python agent.py {CHEMIN_CSV} nom_colonne"
    )

#############################################################################################
#################### Les outils : ce que le code sait faire (le LLM ne voit que la description)
################Chaque fonction ci-dessous appelle une fonction de data_quality_tools.py

FONCTIONS = {
    "describe_colonnes": lambda **kw: DQT.describe_colonnes(df),
    "valeurs_manquantes": lambda **kw: DQT.valeurs_manquantes(df),
    "valeurs_aberrantes": lambda **kw: DQT.valeurs_aberrantes(
        df, exclure=(COLONNE_DEFAUT,)
    ),
    "valeurs_aberrantes_colonne": lambda colonne, **kw: DQT.valeurs_aberrantes(
        df[[colonne]]
    ),
    "modalites_colonne": lambda colonne, **kw: DQT.modalites_colonne(df, colonne),
    "taux_de_defaut": lambda **kw: DQT.taux_de_defaut(df, COLONNE_DEFAUT),
    "defaut_par_variable": lambda colonne, **kw: DQT.defaut_par_variable(
        df, COLONNE_DEFAUT, colonne
    ),
}


def _outil(nom, description, params=None, optionnels=()):
    """Construit la description d'un outil au format attendu par l'API."""
    params = params or {}
    proprietes = {p: {"type": "string", "description": d} for p, d in params.items()}
    return {
        "type": "function",
        "function": {
            "name": nom,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": proprietes,
                "required": [p for p in params if p not in optionnels],
            },
        },
    }


OUTILS = [
    _outil(
        "describe_colonnes",
        "Donne le nombre de lignes et, pour chaque colonne du jeu de données chargé, "
        "son type et son nombre de valeurs différentes. À utiliser pour une vue d'ensemble.",
    ),
    _outil(
        "valeurs_manquantes",
        "Calcule le pourcentage de valeurs manquantes de chaque colonne du jeu de "
        "données chargé (seules les colonnes concernées sont listées).",
    ),
    _outil(
        "valeurs_aberrantes",
        "Détecte les valeurs aberrantes (méthode de l'écart interquartile) dans TOUTES "
        "les colonnes numériques continues du fichier. Aucun argument.",
    ),
    _outil(
    "valeurs_aberrantes_colonne",
    "Détecte les valeurs aberrantes dans UNE seule colonne précise.",
    {"colonne": "Nom exact de la colonne."},
    ),
    _outil(
        "modalites_colonne",
        "Liste les valeurs différentes d'une colonne et leur fréquence. Utile pour les "
        "colonnes catégorielles ou codées, afin de repérer des valeurs inattendues.",
        {"colonne": "Nom exact de la colonne."},
    ),
    _outil(
        "taux_de_defaut",
        "Calcule le taux de défaut global du portefeuille (colonne de défaut déjà connue).",
    ),
    _outil(
        "defaut_par_variable",
        "Analyse bivariée : taux de défaut selon les valeurs (ou classes) d'une variable. "
        "Utile pour voir quelles variables discriminent le défaut.",
        {"colonne": "Nom exact de la variable explicative."},
    ),
]

######################################################
####################### Consigne donnée au LLM

CONSIGNE = (
    "Tu es un assistant d'audit de données de crédit pour des consultants. "
    "Un fichier est DÉJÀ chargé en mémoire : tu n'y accèdes qu'avec les outils "
    "disponibles. Ne demande JAMAIS de fichier à l'utilisateur. "
    f"Colonnes du fichier : {', '.join(df.columns)}. "
    f"La colonne de défaut est '{COLONNE_DEFAUT}' (1 = défaut, 0 = sain). "
    "Pour toute question sur les données, appelle d'abord les outils adaptés ; "
    "tu peux en enchaîner plusieurs (par exemple pour un audit complet : aperçu, "
    "valeurs manquantes, valeurs aberrantes, puis modalités des colonnes codées). "
    "Ne cite que des chiffres renvoyés par les outils, n'en invente jamais. "
    "Si un outil renvoie une erreur, corrige ton appel ou explique le problème. "
    "Réponds en français, de façon concise : d'abord les constats chiffrés, "
    "puis une courte recommandation."
)


def en_json(resultat):
    """Convertit un résultat en texte JSON (gère aussi les nombres NumPy)."""
    return json.dumps(
        resultat,
        ensure_ascii=False,
        default=lambda o: o.item() if hasattr(o, "item") else str(o),
    )


################################################################################
##############################La boucle de l'agent
#
def poser_question(question):
    messages = [
        {"role": "system", "content": CONSIGNE},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TOURS):

        try:
            reponse = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=OUTILS,
                tool_choice="auto",
                temperature=0,
            )
        except BadRequestError as erreur:
            return f"Le modèle a mal formulé un appel d'outil. Relance ou reformule. Détail : {str(erreur)[:200]}"
        message = reponse.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": a.id,
                        "type": "function",
                        "function": {"name": a.function.name, "arguments": a.function.arguments},
                    }
                    for a in message.tool_calls
                ],
            }
        )

        for appel in message.tool_calls:
            nom = appel.function.name
            try:
                arguments = json.loads(appel.function.arguments or "{}")
                print(f"  [outil] {nom}({arguments})")
                resultat = FONCTIONS[nom](**arguments)
            except Exception as erreur:  # le LLM verra l'erreur et pourra corriger
                resultat = {"erreur": str(erreur)}
                print(f"  [erreur] {erreur}")
            messages.append(
                {"role": "tool", "tool_call_id": appel.id, "content": en_json(resultat)}
            )

    return "Trop d'étapes sans conclusion : reformule ta question plus simplement."


if __name__ == "__main__":
    print(f"Fichier : {CHEMIN_CSV} ({len(df)} lignes, {df.shape[1]} colonnes)")
    print(f"Colonne de défaut : {COLONNE_DEFAUT}  |  Modèle : {MODEL}")
    print("Pose une question (ou 'q' pour quitter). Exemple : Fais un audit complet.\n")
    while True:
        question = input("Question > ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question:
            print("\n" + str(poser_question(question)) + "\n")


