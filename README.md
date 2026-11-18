# CreditRisk Copilot

### L'objectif du projet:
> FR : Agent IA (LLM + tool calling + RAG) qui audite la qualité d'un portefeuille de crédit et répond aux questions sur la documentation réglementaire (IFRS 9, EBA, Bâle), avec deux sources citées.
> ENG : AI agent for credit-risk data-quality audits and regulatory Q&A (LLM, tool calling, RAG), Free stack, no paid API.*



#### Ce que l'agent Fait:

Un consultant pose une question en lien avec l'analyse de données ou la reglementation financière en français et l'agent choisit lui-même le bon outil pour répondre :

- **Audit de données** (Pandas) : aperçu des colonnes, valeurs manquantes, valeurs aberrantes, modalités, taux de défaut global (variable cible) et par variable.
- **Recherche documentaire** (RAG) : retrouve les passages pertinents dans l'orientation EBA/GL/2017/06 et le document BCBS 350, et répond en citant document, page et paragraphe.
- **Refus poli** : pour des questions hors périmètre.

**Confidentialité par conception :** le LLM ne voit jamais les lignes du fichier de données, uniquement des descriptions, des noms de colonnes et des agrégats renvoyés par les outils (les résultats des outils d'analyse de données).



## l'architecture du projet

Question ──► Agent.py (boucle d'agent, max 8 tours)
               │  le LLM (Groq, gpt-oss-120b) choisit un outil de data_quality_tools si la question        
                    concerne les données
               ├─► outils de données (data_quality_tools.py) ──► Pandas sur le CSV
               │       renvoient uniquement des agrégats
               └─► chercher_documentation si c'est une question sur la reglementation (rag_outil.py)
                       │  embedding de la question (multilingual-e5-small)
                       ▼
                   ChromaDB (447 passages) ──► passages sous le seuil de distance
                                               avec document, page, paragraphe

Indexation (une fois) : PDF → extraction PyMuPDF → découpage structurel par paragraphe numéroté et sous-point (+ re-découpage des passages > 1 200 caractères) → embeddings → ChromaDB.


#### Stack technique

 Rôle |	 Outil 
 LLM  |	 `openai/gpt-oss-120b` via l'API Groq (palier gratuit), appelée avec le package `openai` 
 Embeddings  |	 `intfloat/multilingual-e5-small` (sentence-transformers, local) 
 Base vectorielle |	 ChromaDB (local) 
 Extraction PDF |	 PyMuPDF 
 Données  |	 Pandas 


## Données et documents utilisés

- Données : [Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — fichier Excel à télécharger à la main.
- BCBS 350 : [Guidance on credit risk and accounting for expected credit losses](https://www.bis.org/bcbs/publ/d350.htm).
- EBA/GL/2017/06 : orientations sur les pratiques de gestion du risque de crédit et la comptabilisation des pertes de crédit attendues (version française), sur [eba.europa.eu](https://www.eba.europa.eu).


## Lancer le projet

```bash
python -m venv .venvagent
.\.venvagent\Scripts\Activate.ps1          # PowerShell
python -m pip install -r requirements.txt

# Clé Groq gratuite : https://console.groq.com
$env:GROQ_API_KEY = "ta_cle"

# 1. Lecture et Préparation de Données : placer "default of credit card clients.xls" dans le dossier, puis
python data_ingestion.py

# 2. Documents : placer les deux PDF dans docs/, puis indexer (le 1er lancement télécharge le modèle d'embedding)
python indexer_documents.py

# 3. recalibrer le seuil de distance (pour le choix de passage, methode cosine)
python calibrer_seuil.py

# 4. Lancer l'agent
python Agent.py
```

Exemples : `taux de defaut global`, `Fais un audit complet`, `c'est quoi la définition de la PD ?`.


## Évaluation (tests manuels)

| Question | Outil attendu | Outil appelé | Source citée | Verdict |
|---|---|---|---|---|
| Taux de défaut global | `taux_de_defaut` | `taux_de_defaut` | n/a | OK |
| Définition de la PD | `chercher_documentation` | `chercher_documentation` | EBA p. 31 §90, p. 37-38 §109-110 | OK : signale que les passages ne contiennent pas de définition |
| Capitale de l'Australie | aucun | aucun | n/a | OK : refus poli |
| Taux directeur de la BCE aujourd'hui | aucun | aucun | n/a | OK : refus poli |
| Que disent les documents sur le LCR ? | `chercher_documentation` | `chercher_documentation` | n/a | OK : conclut à l'absence d'information (le seuil a laissé passer 3 passages à 0,146) |
| Mon taux de défaut est-il cohérent avec le passage entre stades ? | `taux_de_defaut` puis `chercher_documentation` | les deux, dans l'ordre | EBA section 4.3 §107 | Partiel : outils et séparation des sources OK, mais l'analyse de cohérence va trop loin (voir limites) |





## Erreurs rencontrées et corrections

- **Erreur 400 Groq** (`/colonne expected string, but got null`) : un paramètre optionnel d'outil produisait `null`. Corrigé en séparant en deux outils à paramètres obligatoires.
- **Erreur 413 (limite de 8 000 tokens/minute)** : certains passages faisaient plusieurs milliers de caractères et dépassaient aussi la limite de ~512 tokens du modèle d'embedding. On a corrigé cet erreur par un re-découpage à 1200 caractères avec chevauchement, `TOP_K = 3` et une taille maximale par passage envoyé au LLM.
- **Définition de la PD complétée de mémoire** : le modèle ajoutait des informations et un titre de document absents des passages. On a corrigé par une consigne stricte (répondre uniquement à partir des passages, dire quand ils ne répondent pas).
- **Question hors périmètre traitée** (« capitale de l'Australie ») : ajout d'une règle de périmètre dans la consigne.



## Limites observées et perspectives

- **Calibration sur un petit échantillon** (12 questions) : à élargir.
- **Pas de mémoire entre les questions** : chaque question est indépendante.
- **Données :** jeu UCI taïwanais (2005), non représentatif d'un portefeuille réel. Deux documents seulement dans la base, à enrichir.
- **Le seuil de distance est un filtre grossier.** Les distances E5 sont concentrées (0,09 à 0,26) : sur 12 questions de calibration, l'écart entre le pire cas « dans le périmètre » (0,153) et le meilleur « hors périmètre » (0,158) est de 0,005. Une question du domaine absente des documents (LCR) passe le seuil. Le vrai garde-fou est la consigne du LLM.
- **Palier gratuit Groq :** limites de débit ; attendre quelques secondes entre des questions documentaires enchaînées.



## Structure du dépôt : rôle de chaque fichier

| `Agent.py` | boucle d'agent, tool calling ,outils, consigne |
| `data_quality_tools.py` | fonctions d'audit Data Quality |
| `data_ingestion.py` | préparation des données en CSV depuis le fichier Excel téléchargé |
| `Rag_extraction_chunking.py` | extraction PDF et découpage structurel |
| `Rag_embedding.py` | embeddings (préfixes `query:` / `passage:`) |
| `vector_store.py` | ChromaDB (création corpus documentaire) |
| `indexer_documents.py` | indexation des PDF de `docs/` |
| `retrieval.py` | recherche des passages |
| `rag_outil.py` | outil `chercher_documentation` |
| `calibrer_seuil.py` | calibration du seuil de distance |
