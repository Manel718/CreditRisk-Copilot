import json
import os

import pandas as pd
from openai import OpenAI

##Connexion au LLM (Groq)
client= OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

##choix du modèle 
MODEL="openai/gpt-oss-120b"

test_data=pd.DataFrame({
    "client_id":[1,2,3,4,5,6,7,8,9,10],
    "revenu":[2500,None,3000,1500,6000,None,None,3200,None,4000],
    "montant_credit":[10000,5000,None,60000,10000,None,15000,None,9000,18000],
    "Defaut":[0,1,1,0,1,0,0,1,0,1],
})

##Tools
def valeurs_manquantes():
    pourcent=test_data.isna().mean()*100
    return(pourcent.round(1).to_dict())


## description outil
outils=[
    {
        "type":"function",
        "function":{
            "name":"valeurs_manquantes",
            "description":(
                "calcule le pourcentage des valeurs manquantes"
                "dans chaque colonne de jeu de données."
                "du jeu de données déjà chargé. Aucun fichier ni argument à fournir."
            ),
            "parameters":{"type":"object","properties":{}},
        }

    }
]

## la conversation
messages=[
    {
        "role":"system",
        "content":("Tu es un assistant d'analyse de donnees bancaires. "
        "Un jeu de données est DÉJÀ chargé en mémoire : tu n'y accèdes "
        "Utilise les outils disponibles quand c'est utile."
        "Reponds en français."),

    },
    {"role":"user",
     "content":"Y a t-il des problèmes de qualité dans mes données ?"
     }

]

print(outils)
print(messages)
##Premier tool calling
reponse=client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=outils,

)
message=reponse.choices[0].message

if message.tool_calls:
    print("le llm demande à utiliser l'outil")
    messages.append(message)

    for appel in message.tool_calls:
        nom= appel.function.name
        print("f -> outil demande : {nom}")
        if nom=="valeurs_manquantes":
            resultat=valeurs_manquantes()
        else :
            resultat={"erreur": "f -> outil non trouvé: {nom}"}
        print(f" -> resultat : {resultat}")

        messages.append({
            "role":"tool",
            "tool_call_id":appel.id,
            "content":json.dumps(resultat),
        })
    
    ## deuxième call : résultat
    reponse_finale=client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=outils,
    )
    
    print("\nRéponse finale : ")
    print(reponse_finale.choices[0].message.content)
else : 
    print(message.content)



