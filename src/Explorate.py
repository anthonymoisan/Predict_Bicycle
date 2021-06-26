import matplotlib.pyplot as plt
import seaborn as sns
from Input import Input
import pandas as pd
from matplotlib.pyplot import figure

# Classe permettant d'avoir des informations sur les variables unidimensionnelles mais aussi multivariées
class Explorate:

    def __init__(self,input):
        self.input = input

    def __histogramme(self, var, title, xlabel):
        self.input.df[var].value_counts().sort_index().plot(kind="bar")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.show()
    
    def __showMonth(self):
        self.__histogramme("month","Distribution des dates par mois", "Month")

    def __showDay(self):
        self.__histogramme("day","Distribution des dates par jour", "Jour")

    def __showHour(self):
        self.__histogramme("hour","Distribution des dates par heure", "Heure")

    def __describe(self):
        print(self.input.df.describe())

    def __showSeason(self):
        self.__histogramme("season","Distribution des saisons","season")


    def __showVariableNumerique(self,var,title,xlabel):
        self.input.df[var].hist()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.show()
    
    def __variableAExpliquer(self):
        self.__showVariableNumerique("count", "Distribution du nombre de locations de vélos", "# nombre de vélos loués")
    
    def __variableTemperature(self):
        self.__showVariableNumerique("temp", "Distribution de la température", "# Temp en degré celsius")
    
    def __variableATemperature(self):
        self.__showVariableNumerique("atemp", "Distribution de la température ressentie", "# Temp en degré celsius")
    
    def __variableHumidité(self):
        self.__showVariableNumerique("humidity", "Distribution de l'humidité", "# Taux")

    def __variableVent(self):
        self.__showVariableNumerique("windspeed", "Distribution de la vitesse du vent", "# km/h")   
    
    def __modifyWeather(self):
        self.input.df["weather"].replace(
        to_replace=["Légère pluie ou neige", "Fortes averses ou neiges"],
        value='Pluie ou neige',
        inplace=True
        )
        #il faut remettre en catégorielle la variable suite aux remplacements des valeurs et on vérifie les proportions
        self.input.df["weather"] = pd.Categorical(self.input.df["weather"], ordered=True)
        self.input.df["weather"].value_counts(normalize=True)

    def unidimensionnel(self):
        print("\n--- Visualisation des mois")
        self.__showMonth()
        print("coucou")
        print("\n--- Visualisation des jours")
        self.__showDay()
        print("\n--- Visualisation des heures")
        self.__showHour()
        print("\n--- Visualisation des données numériques")
        self.__describe()
        print("\n--- Visualisation de la variable à expliquer")
        self.__variableAExpliquer()
        print("\n--- Visualisation de la variable température")
        self.__variableTemperature()
        print("\n--- Visualisation de la variable température ressentie")
        self.__variableATemperature()
        print("\n--- Visualisation de l'humidité'")
        self.__variableHumidité()
        print("\n--- Visualisation de la variable vent")
        self.__variableVent()
        print("\n--- Visualisation des saisons")
        self.__showSeason()
        print("\n--- Agrégation des modalités Légère pluie ou neige et Fortes averses ou neiges")
        self.__modifyWeather()

    def __biVariee(self,var):
        sns.catplot(x=var, y="count", kind="box", data=self.input.df)
        plt.show()
    
    def __heatMap(self):
        sns.heatmap(self.input.df.corr(), cmap="YlOrRd")
        plt.title("Corrélations des variables continues")
        plt.show()

    def __boxPlot(self,var,title):
        fig = plt.figure()
        fig.set_size_inches(8,12)
        sns.boxplot(data=self.input.df, x=var,y="count", palette="Set2").set_title(title)
        plt.show()

    def biVariee(self):
        print("\n--- Annee vs nombre de vélos")
        self.__biVariee("year")
        print("\n--- Mois vs nombre de vélos")
        self.__biVariee("month")
        print("\n--- Heure vs nombre de vélos")
        self.__biVariee("hour")
        print("\n--- HeatMap")
        self.__heatMap()
        print("\n--- Boxplot of count vs weather")
        self.__boxPlot("weather","Boxplot of count vs weather")
        print("\n--- Boxplot of count vs season")
        self.__boxPlot("season","Boxplot of count vs season")
        print("\n--- Boxplot of count vs workingday")
        self.__boxPlot("workingday","Boxplot of count vs workingday")
        print("\n--- Boxplot of count vs holiday")
        self.__boxPlot("holiday","Boxplot of count vs holiday")
    
    def multivariee(self):
        print("\n--- Temp / Humidité / #nombre de vélos")
        sns.relplot(x="temp", y="humidity", size="count", sizes=(15, 100), data=self.input.df)
        plt.show()
        print("\n--- Hour / workingday / #nombre de vélos")
        sns.catplot(x="hour", y="count", hue="workingday",
            kind="bar",
            data=self.input.df)
        plt.show()

        print("\n--- Hour / workingday / #nombre de vélos / dayofweek")
        self.input.df["dayofweek"] = self.input.df["datetime"].apply(lambda x : x.weekday())
        self.input.df["dayofweek"] = pd.Categorical(self.input.df["dayofweek"]).rename_categories({0 : "Monday",1 : "Tuesday",2 : "Wednesday",3 : "Thursday",4 : "Friday", 5: "Saturday",6 : "Sunday"})
        sns.catplot(x="hour", y="count", hue="workingday",
            col="dayofweek", aspect=.9,
            kind="swarm", data=self.input.df)
        plt.show()

        print("\n--- Hour / #nombre de vélos / season")
        sns.pointplot(x="hour", y="count", hue="season", join=True, data=self.input.df);
        plt.show()