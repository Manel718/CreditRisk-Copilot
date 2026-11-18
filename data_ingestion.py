
################### Télécharge "Default of Credit Card Clients" (UCI, id=350)

import sys
 
import numpy as np
import pandas as pd
 
FICHIER_EXCEL = "credit_card_default.xls"
 
# Dans ce fichier, la 1re ligne contient des étiquettes (X1, X2 ...) et les vrais
# noms de colonnes sont sur la 2e ligne : c'est pourquoi on écrit header=1.
df = pd.read_excel(FICHIER_EXCEL, header=1)
df = df.drop(columns="ID", errors="ignore")  # l'identifiant client n'est pas utile
df = df.rename(columns={"default payment next month": "default_payment_next_month"})
 
# Vérification : on s'assure que les colonnes sont présents dans le fichiers
attendues = {"LIMIT_BAL", "SEX", "EDUCATION", "AGE", "BILL_AMT1", "default_payment_next_month"}
manquantes = attendues - set(df.columns)
if manquantes:
    sys.exit(
        f"Colonnes introuvables : {sorted(manquantes)}\n"
        f"Colonnes lues dans le fichier : {list(df.columns)}\n"
        "Si les noms ressemblent à X1, X2... essaie header=0 à la place de header=1."
    )
 
df.to_csv("credit_card_default.csv", index=False)
####### la version OG
df.to_csv("credit_card_default.csv", index=False)
print(f"credit_card_default.csv : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("Colonnes :", list(df.columns))

# ajout des valeurs manquantes et des valeurs aberrantes pour tester l'agent 
rng = np.random.default_rng(42)  
sale = df.copy()
for col in ["LIMIT_BAL", "AGE", "BILL_AMT1"]:
    sale[col] = sale[col].astype(float)  

sale.loc[rng.random(len(sale)) < 0.10, "LIMIT_BAL"] = np.nan   
sale.loc[rng.random(len(sale)) < 0.05, "AGE"] = np.nan         
sale.loc[rng.choice(len(sale), 20, replace=False), "BILL_AMT1"] = 50_000_000  
sale.loc[rng.choice(len(sale), 10, replace=False), "AGE"] = 150               

sale.to_csv("credit_card_default_sale.csv", index=False)
print("credit_card_default_sale.csv créé (manquants et aberrantes ajoutés).")