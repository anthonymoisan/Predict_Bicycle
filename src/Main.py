from Input import Input
from Explorate import Explorate
from Modele import LinearRegression
from Modele import Tree
from Modele import RandomForest
import os

if __name__ == '__main__':
    
    root = os.path.dirname(__file__)
    rel_path = "../input/velo.csv"
    filename = os.path.join(root, rel_path)
    print("1-Lecture des données et traitement")
    input = Input(filename)
    input.dealWithData()

    print("\n\n2-Analyses")
    explorate = Explorate(input)
    print("\n->-Unidimensionnel")
    explorate.unidimensionnel()
    print("\n->-Bi Variée")
    explorate.biVariee()
    print("\n->-Multi Variée")
    explorate.multivariee()
    
    print("\n\n3-Modeles")
    print("\n\n->-Regression linéaire")
    regLin = LinearRegression(input)
    regLin.run()

    print("\n\n->-Arbre de décision")
    tree = Tree(input)
    tree.run()
    tree.optimizeMaxDepth()
    
    print("\n\n->-RandomForest")
    randomForest = RandomForest(input)
    randomForest.run()
    randomForest.optimizeParameters()