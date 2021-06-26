from Input import Input
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class Modele(ABC):

    def __init__(self,input):
        #ensemble des variables explicatives
        self.X = input.df.drop(["atemp", "count","datetime","season"], axis = 1)
        #variable à expliquer
        self.y = input.df["count"]
        self.__dealCategorical()
        self.__train_test_split()
        self.__dealNumerical()

    def shape(self):
        print(f"Shape de X : {self.X.shape}")
        print(f"Shape de y : {self.y.shape}")
    
    # encodage des variables explicatives catégorielles à l'aide de variables indicatrices.
    def __dealCategorical(self):
        print("--- Traitement des variables catégorielles")
        categorical_features = self.X.columns[self.X.dtypes == "category"].tolist()
        #print(categorical_features)
        df_dummies =  pd.get_dummies(self.X[categorical_features], drop_first=True)
        self.X = pd.concat([self.X.drop(categorical_features, axis=1), df_dummies], axis=1)
        #print(self.X.head(5))

    # sépartion entre ensemble de train et de test
    def __train_test_split(self):
        print("--- Définition des ensembles Train et Test")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=777)
        """print(f"Shape du X_train : {self.X_train.shape}")
        print(f"Shape du y_train : {self.y_train.shape}")
        print(f"Shape du X_test : {self.X_test.shape}")
        print(f"Shape du y_test : {self.y_test.shape}")"""
    
    # certaines méthodes peuvent être sensibles aux échelles des valeurs numériques. Faire un centrage
    def __dealNumerical(self):
        #print("--- Traitement des variables numériques")
        #numerical_features = self.X.columns[(self.X.dtypes == "int64")].tolist() + self.X.columns[(self.X.dtypes == "float64")].tolist()
        #print(numerical_features)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
    
    def regression_metrics(self, y, y_pred):
        return pd.DataFrame(
        {
            "max_error": metrics.max_error(y_true=y, y_pred=y_pred),
            "mean_absolute_error": metrics.mean_absolute_error(y_true=y, y_pred=y_pred),
            "mean_squared_error": metrics.mean_squared_error(y_true=y, y_pred=y_pred),
            "r2_score": metrics.r2_score(y_true=y, y_pred=y_pred)
        },
        index=[0])
    
    @abstractmethod
    def run(self):
        pass


class LinearRegression(Modele):
    
    def __init__(self,input):
        Modele.__init__(self,input)

    def run(self):
        reg = linear_model.LinearRegression()
        reg.fit(self.X_train_scaled, self.y_train)
        print(f"Score sur le train : {reg.score(self.X_train_scaled,self.y_train)}")
        print(f"Score sur le test : {reg.score(self.X_test_scaled,self.y_test)}")
        coefficients = pd.Series(reg.coef_.flatten(), index=self.X.columns).sort_values(ascending=False)
        print(f"ordonnee à l'origine : {reg.intercept_}")
        coefficients[np.abs(coefficients)>10].plot(kind="bar")
        plt.title("Regression lineaire coefficient")
        plt.ylabel("Coefficient value")
        plt.show()

        y_trainPred = reg.predict(self.X_train_scaled)
        y_testPred = reg.predict(self.X_test_scaled)
        print("Regression metrics for train data")
        print(self.regression_metrics(self.y_train, y_trainPred))
        print("Regression metrics for test data")
        print(self.regression_metrics(self.y_test, y_testPred))

class Tree(Modele):

    def __init__(self,input):
        Modele.__init__(self,input)

    def run(self):
        decisionTree = DecisionTreeRegressor()
        decisionTree.fit(self.X_train_scaled, self.y_train)
        print(f"Score sur le train de l'arbre de décision : {decisionTree.score(self.X_train_scaled,self.y_train)}")
        print(f"Score sur le test de l'arbre de décision : {decisionTree.score(self.X_test_scaled,self.y_test)}")

        y_trainPred = decisionTree.predict(self.X_train_scaled)
        y_testPred = decisionTree.predict(self.X_test_scaled)
        print("Regression metrics with Decision Tree for train data")
        print(self.regression_metrics(self.y_train, y_trainPred))
        print("Regression metrics with Decision Tree for test data")
        print(self.regression_metrics(self.y_test, y_testPred))

    def optimizeMaxDepth(self):
        print("\nOptimisation de la profondeur de l'arbre")
        for depth in range(5,20):
            decisionTreeMaxDepth = DecisionTreeRegressor(max_depth=depth)
            decisionTreeMaxDepth.fit(self.X_train_scaled, self.y_train)
            print(f"Max depth : {depth}")
            print(f"Score sur le train de l'arbre de décision : {decisionTreeMaxDepth.score(self.X_train_scaled,self.y_train)}")
            print(f"Score sur le test de l'arbre de décision : {decisionTreeMaxDepth.score(self.X_test_scaled,self.y_test)}")

class RandomForest(Modele):

    def __init__(self,input):
        Modele.__init__(self,input)

    def run(self):
        nbTree = 100
        print(f"Nombre d'arbres considérés : {nbTree}")
        for depth in [5,10,15,20,30, 40]:
            randomForest = RandomForestRegressor(n_estimators=nbTree, random_state=2, max_depth=depth)
            randomForest.fit(self.X_train_scaled, self.y_train)
            print(f"--- Max depth : {depth}")
            print(f"---------Score sur le train avec RandomForest : {randomForest.score(self.X_train_scaled,self.y_train)}")
            print(f"---------Score sur le test avec RandomForest : {randomForest.score(self.X_test_scaled,self.y_test)}")

    def optimizeParameters(self):
        print("\nOptimisation des paramètres de Random Forest")
        # grille de valeurs
        params = [{"max_depth": [10,15,20], "n_estimators": [100,200,300,500], "max_features": [12, 15, 20, 25]}]
        gridSearchCV = GridSearchCV(RandomForestRegressor(), params, cv=5, n_jobs=-1, return_train_score=True)
        gridSearchCV.fit(self.X_train_scaled, self.y_train)   
        print("Score sur le test : {:.2f}".format(gridSearchCV.score(self.X_test_scaled,self.y_test)))
        print("Best parameters : {}".format(gridSearchCV.best_params_))
        print("Best cross-validation score : {:.2f}".format(gridSearchCV.best_score_))
        y_testPred = gridSearchCV.best_estimator_.predict(self.X_test_scaled)
        y_trainPred = gridSearchCV.best_estimator_.predict(self.X_train_scaled)
        print("Regression metrics pour la forêt aléatoire optimisée for test data")
        print(self.regression_metrics(self.y_train, y_trainPred))
        print("Regression metrics pour la forêt aléatoire optimisée for test data")
        print(self.regression_metrics(self.y_test, y_testPred))