import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# permet de savoir si la ligne fait partie du train ou du test
def retourneEnsembleTrainTest(y_train,y_test):
    #y_train et y_test sont des séries pandas avec index correspondant à la ligne dans y et val le nombre de vélos
    listValue = [None] * (len(y_train)+len(y_test))
    
    for index in y_train.index:
        listValue[index] = 1
       
    for index in y_test.index:
        listValue[index] = 0

    for val in listValue:
        if(not((val==0) or (val==1))):
            print("Error")

    return listValue


def regression_metrics(y, y_pred):
        return pd.DataFrame(
        {
            "max_error": metrics.max_error(y_true=y, y_pred=y_pred),
            "mean_absolute_error": metrics.mean_absolute_error(y_true=y, y_pred=y_pred),
            "mean_squared_error": metrics.mean_squared_error(y_true=y, y_pred=y_pred),
            "r2_score": metrics.r2_score(y_true=y, y_pred=y_pred)
        },
        index=[0])

if __name__ == '__main__':

    root = os.path.dirname(__file__)
    rel_path = "../input/velo.csv"
    filename = os.path.join(root, rel_path)

    #lecture
    dataset = pd.read_csv(filename)
    
    #travail sur les dates
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%Y%m%d %H:%M:%S')       
    dataset['month']=dataset["datetime"].apply(lambda x: x.month)
    dataset['day'] = dataset["datetime"].apply(lambda x: x.day)
    dataset['year'] = dataset["datetime"].apply(lambda x: x.year)
    dataset['hour'] = dataset["datetime"].apply(lambda x : x.hour)

    #variables catégorielles
    dataset["season"] = pd.Categorical(dataset["season"], ordered=True).rename_categories({1:'printemps', 2:'été', 3:'automne', 4:'hiver' })
    dataset["holiday"] = pd.Categorical(dataset["holiday"], ordered=False)
    dataset["workingday"] = pd.Categorical(dataset["workingday"], ordered=False)
    dataset["weather"] = pd.Categorical(dataset["weather"], ordered=True).rename_categories({1: 'Dégagé à nuageux', 2 : 'Brouillard', 3 : 'Légère pluie ou neige', 4 : 'Fortes averses ou neiges' })
    dataset["month"] = pd.Categorical(dataset["month"], ordered=True)
    dataset["day"] = pd.Categorical(dataset["day"], ordered=True)
    dataset["year"]= pd.Categorical(dataset["year"], ordered=True)
    dataset["hour"]= pd.Categorical(dataset["hour"], ordered=True)

    #analyse exploratoire a permis de voir qu'une modalité était peu représentée
    dataset["weather"].replace(
    to_replace=["Légère pluie ou neige", "Fortes averses ou neiges"],
    value='Pluie ou neige',
    inplace=True
    )
    #il faut remettre en catégorielle la variable suite aux remplacements des valeurs et on vérifie les proportions
    dataset["weather"] = pd.Categorical(dataset["weather"], ordered=True)

    #supprimer des variables liées
    dataset.drop(["casual", "registered"], axis=1)

    #suppression d'autres variables pour construire la matrice X des variables explicatives
    X = dataset.drop(["atemp", "count","datetime","season"], axis = 1)
    #la variable à expliquer
    y = dataset["count"]

    #traitement des variables catégrorielles avec dummification des variables 
    categorical_features = X.columns[X.dtypes == "category"].tolist()
    dataset_dummies =  pd.get_dummies(X[categorical_features], drop_first=True)
    X = pd.concat([X.drop(categorical_features, axis=1), dataset_dummies], axis=1)

    #définition des ensembles d'apprentissage et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

    listTrainOrTest = retourneEnsembleTrainTest(y_train, y_test)

    #centrage des variables numériques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Meilleur modèle est RandomForest entre régression linéaire et arbre de décision et on met les paramètres optimisées suite à GridSearch
    nbTree = 500
    depth = 20
    feature = 25
    randomForest = RandomForestRegressor(n_estimators=nbTree, random_state=2, max_depth=depth, max_features=feature)
    randomForest.fit(X_train_scaled, y_train)

    #insertion à la dernière position du dataframe les prédictions de notre modèle sur la matrice X centrée-réduite
    dataset.insert(len(dataset.columns),"Predictions", randomForest.predict(scaler.transform(X)))
    dataset.insert(len(dataset.columns), "Train or Test", listTrainOrTest )
    print(dataset.tail(5))
    y_testPred = randomForest.predict(X_test_scaled)
    y_trainPred = randomForest.predict(X_train_scaled)
    print("Regression metrics pour la forêt aléatoire optimisée for train data")
    print(regression_metrics(y_train, y_trainPred))
    print("Regression metrics pour la forêt aléatoire optimisée for test data")
    print(regression_metrics(y_test, y_testPred))


''' Script Python à insérer dans POWER BI. On enlève la lecture qui est réalisé par Power BI et les métriques et trace
# 'dataset' contient les données d'entrée pour ce script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

#travail sur les dates
dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%Y%m%d %H:%M:%S')       
dataset['month']=dataset["datetime"].apply(lambda x: x.month)
dataset['day'] = dataset["datetime"].apply(lambda x: x.day)
dataset['year'] = dataset["datetime"].apply(lambda x: x.year)
dataset['hour'] = dataset["datetime"].apply(lambda x : x.hour)

#variables catégorielles
dataset["season"] = pd.Categorical(dataset["season"], ordered=True).rename_categories({1:'printemps', 2:'été', 3:'automne', 4:'hiver' })
dataset["holiday"] = pd.Categorical(dataset["holiday"], ordered=False)
dataset["workingday"] = pd.Categorical(dataset["workingday"], ordered=False)
dataset["weather"] = pd.Categorical(dataset["weather"], ordered=True).rename_categories({1: 'Dégagé à nuageux', 2 : 'Brouillard', 3 : 'Légère pluie ou neige', 4 : 'Fortes averses ou neiges' })
dataset["month"] = pd.Categorical(dataset["month"], ordered=True)
dataset["day"] = pd.Categorical(dataset["day"], ordered=True)
dataset["year"]= pd.Categorical(dataset["year"], ordered=True)
dataset["hour"]= pd.Categorical(dataset["hour"], ordered=True)

#analyse exploratoire a permis de voir qu'une modalité était peu représentée
dataset["weather"].replace(
    to_replace=["Légère pluie ou neige", "Fortes averses ou neiges"],
    value='Pluie ou neige',
    inplace=True
    )
    
#il faut remettre en catégorielle la variable suite aux remplacements des valeurs et on vérifie les proportions
dataset["weather"] = pd.Categorical(dataset["weather"], ordered=True)

#supprimer des variables liées
dataset.drop(["casual", "registered"], axis=1)

#suppression d'autres variables pour construire la matrice X des variables explicatives
X = dataset.drop(["atemp", "count","datetime","season"], axis = 1)
#la variable à expliquer
y = dataset["count"]

#traitement des variables catégrorielles avec dummification des variables 
categorical_features = X.columns[X.dtypes == "category"].tolist()
dataset_dummies =  pd.get_dummies(X[categorical_features], drop_first=True)
X = pd.concat([X.drop(categorical_features, axis=1), dataset_dummies], axis=1)

#définition des ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777)

#centrage des variables numériques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Meilleur modèle est RandomForest entre régression linéaire et arbre de décision et on met les paramètres optimisées suite à GridSearch
nbTree = 500
depth = 20
feature = 25
randomForest = RandomForestRegressor(n_estimators=nbTree, random_state=2, max_depth=depth, max_features=feature)
randomForest.fit(X_train_scaled, y_train)

#insertion à la dernière position du dataframe les prédictions de notre modèle sur la matrice X centrée-réduite
dataset.insert(len(dataset.columns),"Predictions", randomForest.predict(scaler.transform(X)))
    
'''
        