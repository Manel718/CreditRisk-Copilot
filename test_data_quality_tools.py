import pandas as pd
import data_quality_tools as DQT


##### test une variable ######

df=pd.DataFrame({
    "revenu":[1000,2000,5000,None,10000000,2,3000,6000,4000,8000,None],
    "sexe":[1,1,1,2,None,1,2,2,2,2,2],
    "defaut":[1,1,1,0,0,0,0,0,1,1,1],
    }
)

##### test description des colonnes #######
res = DQT.describe_colonnes(df)
assert res["nb_lignes"] == 11
assert res["nb_colonnes"] == 3
print("OK apercu_colonnes")

####### valeurs_manquantes ######
res = DQT.valeurs_manquantes(df)
assert res["colonne_avec_val_manquants"] == {"revenu": 18.2,"sexe":9.1}   
print("OK  valeurs_manquantes")


####### valeurs_aberrantes ##########
res = DQT.valeurs_aberrantes(df, exclure=("defaut"))
assert list(res) == ["revenu"]            
assert res["revenu"]["nb_aberrantes"] == 1  # seulement le revenu de 10 millions
print("OK  valeurs_aberrantes")
 
######## modalites_colonne ###########
res = DQT.modalites_colonne(df, "sexe")
assert res["modalites"] == {"1.0": 4, "2.0": 6}
print("OK  modalites_colonne")
 
###### taux_de_defaut global ###############
res = DQT.taux_de_defaut(df, "defaut")
assert res["nb_defauts"] == 6
assert res["taux_defaut_pct"] == 54.5
print("OK  taux_de_defaut")
 
####### defaut_par_variable #############
res = DQT.defaut_par_variable(df, "defaut", "sexe")
assert res["1.0"]["taux_defaut_pct"] == 75.0
assert res["2.0"]["taux_defaut_pct"] == 50.0 
print("OK  defaut_par_variable")