
from retrieval import rechercher_passages
from vector_store import creer_collection



###########jeu d'évaluation (références pertinentes attendues)


BASEL = "201512-guidelines-guidance-credit-risk-and-accounting-expected-credit-losses.pdf"
EBA = "orientations-gl-eba-2017-06.pdf"

JEU_EVALUATION = [
    {
        "question": "How should banks validate their ECL models?",
        "passages_attendus": [
            (BASEL, "60"),
            (BASEL, "61")
        ]
    },
    {
        "question": "What are the responsibilities of senior management in ECL governance?",
        "passages_attendus": [
            (BASEL, "25")
        ]
    },
    {
        "question": "What factors should banks consider when estimating ECL?",
        "passages_attendus": [
            (BASEL, "20"),
            (BASEL, "31"),
            (BASEL, "38"),
            (BASEL, "56"),
            (BASEL, "65")
        ]
    },
    {
        "question": "How should forward-looking information be incorporated into ECL estimates?",
        "passages_attendus": [
            (BASEL, "20"),
            (BASEL, "56"),
            (BASEL, "64"),
            (BASEL, "65"),
            (BASEL, "79")
        ]
    },
    {
        "question": "What is the role of PD and LGD in ECL measurement?",
        "passages_attendus": [
            (BASEL, "31"),
            (BASEL, "59")
        ]
    }
]



#######Identification des passages

def identifier_passage(passage):

    metadata = passage["metadata"]

    document = metadata.get("document")
    paragraphe = str(metadata.get("paragraphe", ""))

    return (document, paragraphe)



######Calcul des métriques

def calculer_metriques(passages_retrouves, passages_attendus):

    cles_retrouvees = {
        identifier_passage(p)
        for p in passages_retrouves
    }

    cles_attendues = set(passages_attendus)

    intersection = cles_retrouvees.intersection(cles_attendues)

    hit_rate = 1 if intersection else 0

    recall = (
        len(intersection) / len(cles_attendues)
        if cles_attendues else 0
    )

    return hit_rate, recall, intersection



######### Evaluation du retrieval

def evaluer_retrieval(collection, jeu_evaluation, top_k=5):

    resultats_evaluation = []

    for test in jeu_evaluation:

        question = test["question"]
        passages_attendus = test["passages_attendus"]

        print("\n" + "=" * 70)
        print(f"QUESTION : {question}")
        print("=" * 70)

        # Recherche globale dans tous les documents
        passages = rechercher_passages(
            collection,
            question,
            top_k=top_k
        )

        # Calcul des métriques sans filtrer par document
        hit_rate, recall, retrouves = calculer_metriques(
            passages,
            passages_attendus
        )

        resultats_evaluation.append({
            "question": question,
            "hit_rate": hit_rate,
            "recall": recall,
            "passages_retrouves": retrouves,
            "passages_attendus": passages_attendus
        })

        print("\nPassages attendus :")
        for document, paragraphe in passages_attendus:
            print(f"- {document} | paragraphe {paragraphe}")

        print("\nPassages retrouvés parmi les références attendues :")
        for document, paragraphe in sorted(retrouves):
            print(f"- {document} | paragraphe {paragraphe}")

        print(f"\nHit Rate@{top_k} : {hit_rate}")
        print(f"Recall@{top_k} : {recall:.2%}")

    return resultats_evaluation

######## Métrique précision

def calculer_precision(passages_retrouves, passages_attendus):
    references_retrouvees = {
        identifier_passage(p)
        for p in passages_retrouves
    }

    references_attendues = set(passages_attendus)

    if not references_retrouvees:
        return 0.0

    pertinents = references_retrouvees.intersection(
        references_attendues
    )

    return len(pertinents) / len(references_retrouvees)


############## Synthèse globale

def afficher_synthese(resultats, top_k=5):

    if not resultats:
        print("Aucun résultat à évaluer.")
        return

    moyenne_hit_rate = sum(
        r["hit_rate"] for r in resultats
    ) / len(resultats)

    moyenne_recall = sum(
        r["recall"] for r in resultats
    ) / len(resultats)

    print("\n" + "=" * 70)
    print("SYNTHÈSE DE L'ÉVALUATION")
    print("=" * 70)

    for i, resultat in enumerate(resultats, start=1):
        print(f"\nQuestion {i} : {resultat['question']}")
        print(f"Hit Rate@{top_k} : {resultat['hit_rate']}")
        print(f"Recall@{top_k} : {resultat['recall']:.2%}")

    print("\n" + "-" * 70)
    print(f"Nombre de questions : {len(resultats)}")
    print(f"Hit Rate@{top_k} moyen : {moyenne_hit_rate:.2%}")
    print(f"Recall@{top_k} moyen : {moyenne_recall:.2%}")
    print("=" * 70)


############## Exécution

if __name__ == "__main__":

    collection = creer_collection()

    print(f"Nombre de documents dans ChromaDB : {collection.count()}")

    resultats = evaluer_retrieval(
        collection,
        JEU_EVALUATION,
        top_k=5
    )

    afficher_synthese(resultats, top_k=5)
