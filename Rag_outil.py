"""
Outil de l'agent : chercher_documentation

"""

SEUIL_DISTANCE = 0.17   # distance cosinus maximale pour garder un passage : à calibrer (calibrer_seuil.py)
TOP_K = 3               # nombre de passages demandés à ChromaDB
MAX_CAR_PASSAGE = 1500  # taille maximale d'un passage envoyé au LLM (limite de tokens Groq)

_collection = None      # la collection est ouverte une seule fois, au premier appel

###### Renvoie les passages pertinents avec leurs sources, ou un message si rien n'est trouvé.
def chercher_documentation(question):
    
    # Ces imports sont faits ici et non en haut du fichier : le modèle d'embedding
    # (lourd) ne se charge qu'à la première question sur la documentation.
    from retrieval import rechercher_passages
    from vector_store import creer_collection

    global _collection          # on modifie la variable définie en haut du fichier
    if _collection is None:
        _collection = creer_collection()
    if _collection.count() == 0:
        return {"erreur": "La base documentaire est vide : lance d'abord l'indexation des documents."}

    passages = rechercher_passages(_collection, question, top_k=TOP_K)
    pertinents = [p for p in passages if p["distance"] <= SEUIL_DISTANCE]
    if passages:
        print(f"  [rag] {len(pertinents)} passage(s) retenu(s), meilleure distance {passages[0]['distance']:.3f}")

    if not pertinents:
        return {
            "message": "Aucun passage pertinent trouvé dans la documentation pour cette question.",
            "meilleure_distance": round(passages[0]["distance"], 3) if passages else None,
        }

    return {
        "passages": [
            {
                "source": p["metadata"].get("document"),
                "page": p["metadata"].get("page_debut"),
                "section": p["metadata"].get("section"),
                "paragraphe": p["metadata"].get("paragraphe"),
                "texte": p["texte"][:MAX_CAR_PASSAGE],
            }
            for p in pertinents
        ]
    }

