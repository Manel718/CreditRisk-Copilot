
from pathlib import Path

from Rag_extraction_chunking import extraire_pages, decouper_document
from Rag_embedding import generer_embeddings
from vector_store import (
    creer_collection,
    enregistrer_passages,
    verifier_collection
)



# CONFIGURATION
DOSSIER_DOCS = Path("docs")



#### EXTRACTION ET CHUNKING

def traiter_documents():

    tous_les_passages = []

    fichiers_pdf = list(DOSSIER_DOCS.glob("*.pdf"))

    if not fichiers_pdf:
        raise FileNotFoundError(
            f"Aucun fichier PDF trouvé dans {DOSSIER_DOCS.resolve()}"
        )

    print(f"\nNombre de PDF trouvés : {len(fichiers_pdf)}")

    for fichier in fichiers_pdf:

        print("\n" + "=" * 60)
        print(f"Traitement du document : {fichier.name}")
        print("=" * 60)

        # Découpage du document PDF en chunks
        passages = decouper_document(str(fichier))

        # Normalisation de la structure des chunks
        for passage in passages:

            # Conservation des métadonnées existantes
            metadata = passage.get("metadata", {})

            # Si les métadonnées sont directement dans le dictionnaire
            if not metadata:
                metadata = {
                    cle: valeur
                    for cle, valeur in passage.items()
                    if cle not in ("texte", "id", "metadata")
                }

            # ChromaDB accepte les métadonnées scalaires
            metadata = {
                cle: valeur
                for cle, valeur in metadata.items()
                if isinstance(valeur, (str, int, float, bool))
            }

            # Ajout du nom du document source
            metadata["document"] = fichier.name

            tous_les_passages.append({
                "id": f"chunk_{len(tous_les_passages):04d}",
                "texte": passage["texte"],
                "metadata": metadata
            })

        print(f"Chunks extraits : {len(passages)}")

    print("\n" + "=" * 60)
    print("BILAN DE L'EXTRACTION")
    print("=" * 60)

    print(f"Nombre total de chunks : {len(tous_les_passages)}")

    return tous_les_passages



#### GÉNÉRATION DES EMBEDDINGS
def generer_vecteurs(tous_les_passages):

    print("\n" + "=" * 60)
    print("GÉNÉRATION DES EMBEDDINGS")
    print("=" * 60)

    embeddings = generer_embeddings(tous_les_passages)

    print(f"Nombre de passages : {len(tous_les_passages)}")
    print(f"Nombre d'embeddings : {len(embeddings)}")
    print(f"Dimensions : {embeddings.shape}")

    return embeddings



###### STOCKAGE DANS CHROMADB
def stocker_dans_chromadb(tous_les_passages, embeddings):

    print("\n" + "=" * 60)
    print("STOCKAGE DANS CHROMADB")
    print("=" * 60)

    # Réinitialisation de la collection
    collection = creer_collection(reinitialiser=True)

    # Enregistrement des chunks et embeddings
    enregistrer_passages(
        collection,
        tous_les_passages,
        embeddings
    )

    # Vérification
    verifier_collection(collection)

    nombre = collection.count()

    print("\nNombre final de documents :", nombre)

    if nombre != len(tous_les_passages):
        raise ValueError(
            "Le nombre de documents enregistrés ne correspond "
            "pas au nombre de chunks générés."
        )

    print("\nIndexation terminée avec succès !")

    return collection



##### PIPELINE PRINCIPAL
def main():

    # Étape 1 : extraction et chunking
    tous_les_passages = traiter_documents()

    if not tous_les_passages:
        raise ValueError("Aucun chunk n'a été généré.")

    # Étape 2 : embeddings
    embeddings = generer_vecteurs(tous_les_passages)

    # Étape 3 : stockage
    collection = stocker_dans_chromadb(
        tous_les_passages,
        embeddings
    )

    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ")
    print("=" * 60)

    print("Total des chunks :", len(tous_les_passages))
    print("Total dans ChromaDB :", collection.count())


####Main 
if __name__ == "__main__":
    main()
