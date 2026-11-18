
import chromadb



#### CONFIGURATION

CHEMIN_BASE = "./chroma_db"
NOM_COLLECTION = "credit_risk"



#### CONNEXION À CHROMADB

client = chromadb.PersistentClient(path=CHEMIN_BASE)



# Crée ou récupère la collection ChromaDB

def creer_collection(reinitialiser=False):
    
    ### reinitialiser=True Supprime l'ancienne collection et en crée une nouvelle.

    if reinitialiser:
        try:
            client.delete_collection(name=NOM_COLLECTION)
            print("Ancienne collection supprimée.")
        except Exception:
            print("Aucune ancienne collection à supprimer.")

        collection = client.create_collection(
            name=NOM_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        print("Nouvelle collection créée.")

    else:
        collection = client.get_or_create_collection(
            name=NOM_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        print("Collection récupérée ou créée.")

    return collection


##### ENREGISTREMENT DES CHUNKS, LEURS EMBEDDINGS, LEURS METADONNEES DANS ChromaDB

def enregistrer_passages(collection, passages, embeddings):

    if len(passages) != len(embeddings):
        raise ValueError(
            f"Nombre de passages ({len(passages)}) différent "
            f"du nombre d'embeddings ({len(embeddings)})."
        )

    if not passages:
        print("Aucun passage à enregistrer.")
        return

    ids = [str(p["id"]) for p in passages]

    if len(ids) != len(set(ids)):
        raise ValueError("Les identifiants des chunks ne sont pas uniques.")

    documents = [p["texte"] for p in passages]

    metadatas = [p["metadata"] for p in passages]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"{len(passages)} passages enregistrés dans ChromaDB.")



########## VÉRIFICATION DE LA COLLECTION : Affiche le nombre de passages et quelques exemples

def verifier_collection(collection):

    nombre = collection.count()

    print("\n===== ÉTAT DE LA COLLECTION =====")
    print(f"Nombre total de passages : {nombre}")

    if nombre > 0:
        exemples = collection.peek()

        print("\n===== PREMIERS PASSAGES =====")

        for i, document in enumerate(exemples["documents"]):
            print(f"\nPassage {i + 1}")
            print(document[:300], "...")

            print("Métadonnées :", exemples["metadatas"][i])



# TEST 
if __name__ == "__main__":

    collection = creer_collection()

    verifier_collection(collection)
