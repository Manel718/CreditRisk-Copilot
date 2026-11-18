# Rag_extraction_chunking.py

import re
from pathlib import Path
from collections import Counter

import pymupdf


######### MOTIFS DE RECONNAISSANCE

# Exemple de titre : Principle 2 – Sound ECL methodologies
MOTIF_PRINCIPE = re.compile(
    r"^\s*(Principle\s+\d+\s*[–—-].*)$",
    re.IGNORECASE
)

# Exemple de titre : 4.2 Principes relatifs
MOTIF_SECTION = re.compile(
    r"^\s*(\d+(?:\.\d+)+)\s+(.+)$"
)

# Numéro de paragraphe isolé : 31 ou 31.
MOTIF_PARAGRAPHE_ISOLE = re.compile(
    r"^\s*(\d{1,3})\.?\s*$"
)

# Numéro de paragraphe suivi de son texte : 31. The bank
MOTIF_PARAGRAPHE_INLINE = re.compile(
    r"^\s*(\d{1,3})\.\s+([A-Z].*)$"
)

# Sous-points alphabétiques : (a), a), (b)
MOTIF_ALPHA = re.compile(
    r"^\s*\(?([a-z])\)?[.)]?\s*$",
    re.IGNORECASE
)

# Sous-points romains : (i), (ii), (iii)
MOTIF_ROMAIN = re.compile(
    r"^\s*\(([ivxlcdm]+)\)\s*$",
    re.IGNORECASE
)



################## EXTRACTION DU TEXTE PDF ##############################


def extraire_pages(chemin_pdf):
    """
    Extrait le texte de chaque page avec PyMuPDF.
    Conserve le numéro de page et filtre certains éléments
    répétitifs et notes de bas de page.
    """

    document = pymupdf.open(chemin_pdf)
    pages_brutes = []

    for numero_page, page in enumerate(document, start=1):

        blocs = page.get_text("blocks")

        # Trier les blocs verticalement puis horizontalement
        blocs_tries = sorted(
            blocs,
            key=lambda bloc: (bloc[1], bloc[0])
        )

        for bloc in blocs_tries:

            x0, y0, x1, y1, texte = bloc[:5]
            texte = texte.strip()

            if not texte:
                continue

            pages_brutes.append({
                "page": numero_page,
                "y": y0,
                "hauteur_page": page.rect.height,
                "texte": texte
            })

    document.close()

    
    ###################### Identifier les en-têtes répétés ########################
    

    blocs_haut = []

    for bloc in pages_brutes:

        if bloc["y"] < bloc["hauteur_page"] * 0.15:

            texte = " ".join(bloc["texte"].split())
            blocs_haut.append(texte)

    frequences = Counter(blocs_haut)

    nb_pages = max(
        (bloc["page"] for bloc in pages_brutes),
        default=0
    )

    entetes_repetes = {
        texte
        for texte, frequence in frequences.items()
        if frequence >= 2 and nb_pages > 1
    }

    
    ################## Nettoyage des pages

    pages = {}

    for bloc in pages_brutes:

        numero_page = bloc["page"]
        texte = bloc["texte"].strip()
        y = bloc["y"]
        hauteur = bloc["hauteur_page"]

        texte_normalise = " ".join(texte.split())

        # Supprimer les en-têtes répétés
        if texte_normalise in entetes_repetes:
            continue

        # Supprimer les numéros de page isolés
        if re.fullmatch(r"\d{1,3}", texte_normalise):
            continue

        # Supprimer certains pieds de page connus
        if (
            "Guidance on credit risk and accounting "
            "for expected credit losses" in texte
        ):
            continue

        # Supprimer les notes de bas de page
        # uniquement lorsqu'elles sont en bas de page
        # et commencent par un numéro suivi d'un retour ligne.
        est_en_bas = y > hauteur * 0.72

        commence_par_note = re.match(
            r"^\s*\d{1,2}\s*\n",
            texte
        )

        if est_en_bas and commence_par_note:
            continue

        pages.setdefault(numero_page, []).append(texte)

    resultat = []

    for numero_page in sorted(pages):

        texte_page = "\n".join(pages[numero_page])
        resultat.append((numero_page, texte_page))

    return resultat


################################# FONCTIONS UTILITAIRES

def nettoyer_ligne(texte):
    """Nettoie les espaces inutiles."""

    return " ".join(texte.split()).strip()


def lettre_suivante(lettre):
    """Retourne la lettre alphabétique suivante."""

    if lettre is None:
        return None

    return chr(ord(lettre) + 1)


def nouveau_conteneur():
    """Crée un conteneur de texte avec ses pages."""

    return {
        "lignes": [],
        "page_debut": None,
        "page_fin": None
    }


def ajouter_ligne(conteneur, texte, numero_page):
    """Ajoute une ligne et actualise les numéros de page."""

    if not texte:
        return

    conteneur["lignes"].append(texte)

    if conteneur["page_debut"] is None:
        conteneur["page_debut"] = numero_page

    conteneur["page_fin"] = numero_page


def texte_conteneur(conteneur):
    """Reconstitue le texte d'un conteneur."""

    return nettoyer_ligne(" ".join(conteneur["lignes"]))


def creer_chunk(
    source,
    texte,
    page_debut,
    page_fin,
    section=None,
    titre_section=None,
    principe=None,
    paragraphe=None,
    sous_point=None,
    sous_point_romain=None
):
    """Construit un chunk avec ses métadonnées."""

    texte = nettoyer_ligne(texte)

    if not texte:
        return None

    return {
        "source": source,
        "texte": texte,
        "page_debut": page_debut,
        "page_fin": page_fin,
        "section": section,
        "titre_section": titre_section,
        "principe": principe,
        "paragraphe": paragraphe,
        "sous_point": sous_point,
        "sous_point_romain": sous_point_romain
    }


######################## bis. RE-DÉCOUPAGE DES CHUNKS TROP LONGS

MAX_CAR = 1200          # taille maximale d'un chunk, préfixe compris
CHEVAUCHEMENT = 150     # caractères communs entre deux morceaux d'un même chunk


def construire_prefixe(chunk):
    """Reconstruit le préfixe de contexte, au même format que dans ajouter_chunk_paragraphe."""
    if chunk["paragraphe"] is None:      # texte hors paragraphe : pas de préfixe
        return ""

    parties = []
    if chunk["principe"]:
        parties.append(f"Principe : {chunk['principe']}")
    if chunk["section"]:
        parties.append(f"Section : {chunk['section']}")
    if chunk["titre_section"]:
        parties.append(f"Titre de section : {chunk['titre_section']}")
    parties.append(f"Paragraphe : {chunk['paragraphe']}")
    if chunk["sous_point"]:
        parties.append(f"Sous-point : {chunk['sous_point']}")
    if chunk["sous_point_romain"]:
        parties.append(f"Sous-point romain : {chunk['sous_point_romain']}")
    return " | ".join(parties)


def redecouper_longs(chunks, max_car=MAX_CAR, chevauchement=CHEVAUCHEMENT):
    """Coupe les chunks trop longs en morceaux, en remettant le préfixe devant chacun."""
    resultat = []

    for chunk in chunks:
        texte = chunk["texte"]

        if len(texte) <= max_car:
            resultat.append(chunk)
            continue

        prefixe = construire_prefixe(chunk)
        if prefixe and texte.startswith(prefixe):
            corps = texte[len(prefixe):].strip()
        else:
            corps = texte

        taille = max(max_car - len(prefixe) - 1, 400)
        debut = 0

        while debut < len(corps):
            fin = debut + taille

            # Couper sur un espace pour ne pas couper un mot en deux
            if fin < len(corps):
                coupe = corps.rfind(" ", debut + chevauchement + 1, fin)
                if coupe != -1:
                    fin = coupe

            morceau = corps[debut:fin].strip()
            texte_morceau = f"{prefixe} {morceau}" if prefixe else morceau
            resultat.append({**chunk, "texte": texte_morceau})

            if fin >= len(corps):
                break
            debut = fin - chevauchement

    return resultat


################################## CHUNKING STRUCTUREL #########################

def decouper_document(chemin_pdf):

    source = Path(chemin_pdf).name
    pages = extraire_pages(chemin_pdf)

    chunks = []

    # Contexte documentaire
    section_actuelle = None
    titre_section_actuel = None
    principe_actuel = None

    # Paragraphe actuellement traité
    paragraphe = None
    alpha_actuel = None
    romain_actuel = None

    # Dernière lettre alphabétique détectée
    dernier_alpha_label = None

    # Texte hors paragraphe
    texte_hors_paragraphe = nouveau_conteneur()

    
    ############################## Finaliser un paragraphe ############################

    def ajouter_chunk_paragraphe():

        nonlocal paragraphe

        if paragraphe is None:
            return

        contexte = []

        if paragraphe["principe"]:
            contexte.append(
                f"Principe : {paragraphe['principe']}"
            )

        if paragraphe["section"]:
            contexte.append(
                f"Section : {paragraphe['section']}"
            )

        if paragraphe["titre_section"]:
            contexte.append(
                f"Titre de section : {paragraphe['titre_section']}"
            )

        contexte.append(
            f"Paragraphe : {paragraphe['numero']}"
        )

        prefixe = " | ".join(contexte)

        # Introduction du paragraphe
        introduction = texte_conteneur(
            paragraphe["introduction"]
        )

        if introduction:

            chunk = creer_chunk(
                source=source,
                texte=f"{prefixe}\n{introduction}",
                page_debut=paragraphe["introduction"]["page_debut"],
                page_fin=paragraphe["introduction"]["page_fin"],
                section=paragraphe["section"],
                titre_section=paragraphe["titre_section"],
                principe=paragraphe["principe"],
                paragraphe=paragraphe["numero"]
            )

            if chunk:
                chunks.append(chunk)

        # Sous-points
        for alpha in paragraphe["sous_points"]:

            texte_alpha = texte_conteneur(
                alpha["contenu"]
            )

            # Sous-point sans sous-points romains
            if not alpha["romains"]:

                if texte_alpha:

                    chunk = creer_chunk(
                        source=source,
                        texte=(
                            f"{prefixe} | "
                            f"Sous-point : {alpha['label']}\n"
                            f"{texte_alpha}"
                        ),
                        page_debut=alpha["contenu"]["page_debut"],
                        page_fin=alpha["contenu"]["page_fin"],
                        section=paragraphe["section"],
                        titre_section=paragraphe["titre_section"],
                        principe=paragraphe["principe"],
                        paragraphe=paragraphe["numero"],
                        sous_point=alpha["label"]
                    )

                    if chunk:
                        chunks.append(chunk)

            # Sous-point contenant des sous-points romains
            else:

                # Conserver l'introduction du sous-point
                if texte_alpha:

                    chunk = creer_chunk(
                        source=source,
                        texte=(
                            f"{prefixe} | "
                            f"Sous-point : {alpha['label']}\n"
                            f"{texte_alpha}"
                        ),
                        page_debut=alpha["contenu"]["page_debut"],
                        page_fin=alpha["contenu"]["page_fin"],
                        section=paragraphe["section"],
                        titre_section=paragraphe["titre_section"],
                        principe=paragraphe["principe"],
                        paragraphe=paragraphe["numero"],
                        sous_point=alpha["label"]
                    )

                    if chunk:
                        chunks.append(chunk)

                # Ajouter les sous-points romains
                for romain in alpha["romains"]:

                    texte_romain = texte_conteneur(
                        romain["contenu"]
                    )

                    if not texte_romain:
                        continue

                    chunk = creer_chunk(
                        source=source,
                        texte=(
                            f"{prefixe} | "
                            f"Sous-point : {alpha['label']} | "
                            f"Sous-point romain : {romain['label']}\n"
                            f"{texte_romain}"
                        ),
                        page_debut=romain["contenu"]["page_debut"],
                        page_fin=romain["contenu"]["page_fin"],
                        section=paragraphe["section"],
                        titre_section=paragraphe["titre_section"],
                        principe=paragraphe["principe"],
                        paragraphe=paragraphe["numero"],
                        sous_point=alpha["label"],
                        sous_point_romain=romain["label"]
                    )

                    if chunk:
                        chunks.append(chunk)

        paragraphe = None

    ######################## Finaliser le texte hors paragraphe ##########################

    def finaliser_texte_hors_paragraphe():

        nonlocal texte_hors_paragraphe

        texte = texte_conteneur(
            texte_hors_paragraphe
        )

        if texte:

            chunk = creer_chunk(
                source=source,
                texte=texte,
                page_debut=texte_hors_paragraphe["page_debut"],
                page_fin=texte_hors_paragraphe["page_fin"],
                section=section_actuelle,
                titre_section=titre_section_actuel,
                principe=principe_actuel
            )

            if chunk:
                chunks.append(chunk)

        texte_hors_paragraphe = nouveau_conteneur()

    ################################# PARCOURS DES PAGES ###################

    for numero_page, texte_page in pages:

        lignes = texte_page.splitlines()

        for ligne in lignes:

            ligne = nettoyer_ligne(ligne)

            if not ligne:
                continue

            ################ Détection d'un principe

            match_principe = MOTIF_PRINCIPE.match(ligne)

            if match_principe:

                ajouter_chunk_paragraphe()
                finaliser_texte_hors_paragraphe()

                principe_actuel = match_principe.group(1)
                section_actuelle = None
                titre_section_actuel = None

                continue

            ##################### Détection d'une section

            match_section = MOTIF_SECTION.match(ligne)

            if match_section:

                ajouter_chunk_paragraphe()
                finaliser_texte_hors_paragraphe()

                section_actuelle = match_section.group(1)
                titre_section_actuel = match_section.group(2)

                ajouter_ligne(
                    texte_hors_paragraphe,
                    ligne,
                    numero_page
                )

                continue

            
            ################ Détection d'un nouveau paragraphe

            match_para_inline = MOTIF_PARAGRAPHE_INLINE.match(ligne)
            match_para_isole = MOTIF_PARAGRAPHE_ISOLE.match(ligne)

            if match_para_inline or match_para_isole:

                ajouter_chunk_paragraphe()
                finaliser_texte_hors_paragraphe()

                if match_para_inline:

                    numero_para = match_para_inline.group(1)
                    texte_initial = match_para_inline.group(2)

                else:

                    numero_para = match_para_isole.group(1)
                    texte_initial = ""

                paragraphe = {
                    "numero": numero_para,
                    "section": section_actuelle,
                    "titre_section": titre_section_actuel,
                    "principe": principe_actuel,
                    "introduction": nouveau_conteneur(),
                    "sous_points": []
                }

                alpha_actuel = None
                romain_actuel = None
                dernier_alpha_label = None

                if texte_initial:

                    ajouter_ligne(
                        paragraphe["introduction"],
                        texte_initial,
                        numero_page
                    )

                continue

            #### Texte hors paragraphe

            if paragraphe is None:

                ajouter_ligne(
                    texte_hors_paragraphe,
                    ligne,
                    numero_page
                )

                continue

            ############ Détection des sous-points

            match_alpha = MOTIF_ALPHA.match(ligne)
            match_romain = MOTIF_ROMAIN.match(ligne)

            est_alpha = False

            if match_alpha:

                label_candidat = match_alpha.group(1).lower()

                # Premier sous-point du paragraphe
                if dernier_alpha_label is None:

                    est_alpha = True

                # Vérifier la continuité alphabétique :
                # après b, c est considéré comme alphabétique,
                # même si c est également un chiffre romain.
                elif label_candidat == lettre_suivante(
                    dernier_alpha_label
                ):

                    est_alpha = True

            ############ Sous-point alphabétique
            if est_alpha:

                label_alpha = match_alpha.group(1).lower()

                alpha_actuel = {
                    "label": label_alpha,
                    "contenu": nouveau_conteneur(),
                    "romains": []
                }

                paragraphe["sous_points"].append(
                    alpha_actuel
                )

                dernier_alpha_label = label_alpha
                romain_actuel = None

                continue

            ########## Sous-point romain

            if match_romain and alpha_actuel is not None:

                label_romain = match_romain.group(1).lower()

                romain_actuel = {
                    "label": label_romain,
                    "contenu": nouveau_conteneur()
                }

                alpha_actuel["romains"].append(
                    romain_actuel
                )

                continue

            ########### Affectation du texte au bon niveau

            if romain_actuel is not None:

                ajouter_ligne(
                    romain_actuel["contenu"],
                    ligne,
                    numero_page
                )

            elif alpha_actuel is not None:

                ajouter_ligne(
                    alpha_actuel["contenu"],
                    ligne,
                    numero_page
                )

            else:

                ajouter_ligne(
                    paragraphe["introduction"],
                    ligne,
                    numero_page
                )

    ########## FINALISATION

    ajouter_chunk_paragraphe()
    finaliser_texte_hors_paragraphe()

    return redecouper_longs(chunks)