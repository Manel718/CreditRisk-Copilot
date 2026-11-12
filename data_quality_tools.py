
####### packages
import pandas as pd

### description de la base 
def describe_colonnes(df):

    return {
        "nb_lignes": int(len(df)),
        "nb_colonnes": int(df.shape[1]),
        "colonnes":{
            str(c):{
            "type": str(df[c].dtype),
            "valeurs_uniques": int(df[c].nunique()),
        } for c in df.columns

    },
}



#### %valeurs manquantes

def valeurs_manquantes(df):
    pourcent=(df.isna().mean()*100).round(1)
    colonne_avec_val_manquants={str(c):float(p) for c, p in pourcent.items() if p>0}
    return{"nb_lignes": int(len(df)),
            "colonne_avec_val_manquants": colonne_avec_val_manquants,
            "message": "" if colonne_avec_val_manquants else "Aucune valeur manquante.",}
        

#### Valeurs aberrantes 
###Valeurs aberrantes par la méthode de l'écart interquartile (IQR) :
###une valeur est suspecte si elle sort de [Q1 - 1,5*IQR ; Q3 + 1,5*IQR]
seuil_code=3 ## nombre de valeurs uniques 
def valeurs_aberrantes(df,exclure=()):
    
    colonnes = [c for c in df.select_dtypes("number").columns if c not in exclure]
    resultat={}
    for c in colonnes:
        df_drpna=df[c].dropna()
        if df_drpna.nunique() <= seuil_code:
            continue
        q1, q3 = df_drpna.quantile([0.25, 0.75])
        iqr=q3-q1
        borne_basse=q1-1.5*iqr
        borne_haute=1.5*iqr+q3
        nb=int((((df_drpna<borne_basse)|(df_drpna>borne_haute))).sum())
        resultat[str(c)]={
            "nb_aberrantes":nb,
            "pourcentage":round(nb/len(df_drpna)*100,1),
            "bornes":[round(float(borne_basse),1),round(float(borne_haute),1)],
        }
    return(resultat)

#### les valeurs moins frequentes pour les variables catégorielles
def modalites_colonne(df, colonne, max_modalites=4):
    
    comptages = df[colonne].value_counts(dropna=True).head(max_modalites)
    return {
        "colonne": str(colonne),
        "nb_valeurs_differentes": int(df[colonne].nunique(dropna=True)),
        "modalites": {str(k): int(v) for k, v in comptages.items()},
    }


#### taux defaut global 

def taux_de_defaut(df, colonne_defaut):
    """Taux de défaut global, en %. Suppose que la colonne vaut 0 (sain) ou 1 (défaut)."""
    
    valeurs = set(df[colonne_defaut].dropna().unique())
    if not valeurs <= {0, 1}:
        raise ValueError(
            f"La colonne '{colonne_defaut}' doit contenir uniquement 0 et 1, "
            f"mais contient : {sorted(valeurs)[:10]}"
        )
    return {
        "nb_lignes": int(len(df)),
        "nb_defauts": int(df[colonne_defaut].sum()),
        "taux_defaut_pct": round(float(df[colonne_defaut].mean()) * 100, 1),
    }

#### taux de défaut par variables
def defaut_par_variable(df, colonne_defaut, colonne, nb_classes=5):
    
    
    serie = df[colonne]
    if pd.api.types.is_numeric_dtype(serie) and serie.nunique() > seuil_code:
        groupes = pd.qcut(serie, q=nb_classes, duplicates="drop")
    else:
        groupes = serie
 
    tableau = df.groupby(groupes, observed=True)[colonne_defaut].agg(["mean", "count"])
    return {
        str(groupe): {
            "taux_defaut_pct": round(float(ligne["mean"]) * 100, 1),
            "effectif": int(ligne["count"]),
        }
        for groupe, ligne in tableau.iterrows()
    }
 
