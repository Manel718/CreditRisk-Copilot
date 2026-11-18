from sentence_transformers import SentenceTransformer

MODELE_EMBEDDING = "intfloat/multilingual-e5-small"

model = SentenceTransformer(MODELE_EMBEDDING)



def generer_embeddings(passages):

    if not passages:
        return []

    textes = [
        "passage: " + p["texte"]
        for p in passages
    ]

    embeddings = model.encode(
        textes,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


### embedding query
def generer_embedding_question(question):

    texte = "query: " + question

    embedding = model.encode(
        [texte],
        normalize_embeddings=True
    )

    return embedding

