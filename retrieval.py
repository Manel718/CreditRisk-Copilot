
from Rag_embedding import generer_embedding_question

##### Recherche les passages les plus pertinents par rapport à une question utilisateur.

def rechercher_passages(collection, question, top_k=5):
   

    # Générer l'embedding de la question
    embedding_question = generer_embedding_question(question)

    # Recherche dans ChromaDB
    resultats = collection.query(
        query_embeddings=embedding_question.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    

    passages_retrouves = []

    for i in range(len(resultats["documents"][0])):

        passage = {
            "texte": resultats["documents"][0][i],
            "metadata": resultats["metadatas"][0][i],
            "distance": resultats["distances"][0][i]
        }

        passages_retrouves.append(passage)

    return passages_retrouves


##### Affiche les passages retrouvés et leurs distances.
    
def afficher_resultats(passages):
    

    print("\n===== RÉSULTATS DU RETRIEVAL =====")

    for i, passage in enumerate(passages):

        print(f"\n--- Passage {i + 1} ---")
        print("Distance :", round(passage["distance"], 4))
        print("Métadonnées :", passage["metadata"])
        print("Texte :", passage["texte"])


if __name__ == "__main__":

    from vector_store import creer_collection

    collection = creer_collection()
    print("Nombre de documents dans ChromaDB :", collection.count())

    questions = [
    "How should banks validate their ECL models?",
    "What is the role of PD and LGD in ECL measurement?",
    "What are the responsibilities of senior management in ECL governance?"
    ]
    
    for question in questions:
        print(f"\nQUESTION : {question}")
        passages = rechercher_passages(collection, question, top_k=5)
        afficher_resultats(passages)


