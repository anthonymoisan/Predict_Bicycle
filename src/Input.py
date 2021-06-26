import pandas as pd
import numpy as np

#classe permettant de travailler sur les données du problème stocké dans un objet panda
class Input :

    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def info(self):
        self.df.info()

    def showDf(self, size):
        print("taille du jeu de donnees :", self.df.shape)
        print(self.df.head(size))

    def __identifyvariableCategorial(self):        
        self.df["season"] = pd.Categorical(self.df["season"], ordered=True).rename_categories({1:'printemps', 2:'été', 3:'automne', 4:'hiver' })
        self.df["holiday"] = pd.Categorical(self.df["holiday"], ordered=False)
        self.df["workingday"] = pd.Categorical(self.df["workingday"], ordered=False)
        self.df["weather"] = pd.Categorical(self.df["weather"], ordered=True).rename_categories({1: 'Dégagé à nuageux', 2 : 'Brouillard', 3 : 'Légère pluie ou neige', 4 : 'Fortes averses ou neiges' })
        self.df["month"] = pd.Categorical(self.df["month"], ordered=True)
        self.df["day"] = pd.Categorical(self.df["day"], ordered=True)
        self.df["year"]= pd.Categorical(self.df["year"], ordered=True)
        self.df["hour"]= pd.Categorical(self.df["hour"], ordered=True)
        print(self.df.dtypes)

    def __dropVariable(self, listVar) :
        self.df = self.df.drop(listVar, axis=1)

    def __dateTime(self):
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%Y%m%d %H:%M:%S')       
        self.df['month']=self.df["datetime"].apply(lambda x: x.month)
        self.df['day'] = self.df["datetime"].apply(lambda x: x.day)
        self.df['year'] = self.df["datetime"].apply(lambda x: x.year)
        self.df['hour'] = self.df["datetime"].apply(lambda x : x.hour)

    def dealWithData(self):
        print("\n--- Travail sur les dates et feature engeneering ")
        self.__dateTime()

        print("\n--- Identifier les variables catégorielles ")
        self.__identifyvariableCategorial()

        print("\n--- Drop de certaines variables ")
        self.__dropVariable(["casual", "registered"])