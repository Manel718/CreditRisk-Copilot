
######### Calibration du seuil de distance pour l'outil chercher_documentation ###########



from retrieval import rechercher_passages
from vector_store import creer_collection

DANS_LE_PERIMETRE = [
    "Comment évaluer l'augmentation significative du risque de crédit ?",
    "Quelle est la différence entre une provision à 12 mois et à durée de vie ?",
    "Quel est le rôle du conseil d'administration dans la gouvernance des pertes de crédit attendues ?",
    "Comment les banques doivent-elles valider leurs modèles de pertes attendues ?",
    "How should banks validate their ECL models?",
    "What is the role of PD and LGD in ECL measurement?",
    "What are the responsibilities of senior management in ECL governance?",
]

HORS_PERIMETRE = [
    "Quelle est la capitale de l'Australie ?",
    "Comment faire une tarte aux pommes ?",
    "Qui a gagné la Coupe du monde de football en 2018 ?",
    "Explique le fonctionnement d'un moteur diesel.",
    "Quel est le taux directeur de la BCE aujourd'hui ?",   # proche du domaine, mais absent des documents
]


def meilleure_distance(collection, question):
    meilleur = rechercher_passages(collection, question, top_k=3)[0]
    meta = meilleur["metadata"]
    lieu = f"{meta.get('document')} p.{meta.get('page_debut')} §{meta.get('paragraphe')}"
    return meilleur["distance"], lieu


if __name__ == "__main__":
    collection = creer_collection()
    print("Passages indexés :", collection.count())

    resultats = {}
    for nom, questions in [("DANS LE PÉRIMÈTRE", DANS_LE_PERIMETRE), ("HORS PÉRIMÈTRE", HORS_PERIMETRE)]:
        print(f"\n=== {nom} ===")
        distances = []
        for q in questions:
            d, lieu = meilleure_distance(collection, q)
            distances.append(d)
            print(f"  {d:.3f}  {q[:70]}  -> {lieu}")
        resultats[nom] = distances

    pire_dedans = max(resultats["DANS LE PÉRIMÈTRE"])
    meilleur_dehors = min(resultats["HORS PÉRIMÈTRE"])
    print(f"\nPire distance dans le périmètre : {pire_dedans:.3f}")
    print(f"Meilleure distance hors périmètre : {meilleur_dehors:.3f}")
    if pire_dedans < meilleur_dehors:
        print(f"Les deux groupes sont séparés : un seuil autour de {(pire_dedans + meilleur_dehors) / 2:.2f} convient.")
    else:
        print("Les deux groupes se chevauchent : aucun seuil ne sépare tout. ")
