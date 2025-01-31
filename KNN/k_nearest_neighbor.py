## file riguardante il classificatore knn 

"""
Il classificatore knn è un algoritmo che verrà usato per classificare i tumori come benigni o maligni.
- Implementeremo dapprima la metrica di distanza Euclidea per confrontare i campioni 
- Identificheremo i k vicini più vicini a un dato campione 
- Assegneremo una classe in base alla maggioranza dei vicini gestendo anche il caso di pareggio tra classi 
"""""

# knn prende in input features e labels per il training (X_train, y_train) e il parametro k (nummero dei vicini da considerare)
# in output verranno restituite le predizioni y_pred e il modello knn addestrato 

import random 
import pandas as pd 
import numpy as np 
from collections import Counter 

class ClassificatoreKNN: 
    def __init__(self, k = 5):   #inizializzo il classificatore knn e imposto un valore di default per k
        self.k = k
        self.features = None # dati training 
        self.labels = None # etichette training 

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None: 
        """
        con la funzione fit vengono salvati i dati di training 

        INPUT: 
        features (pd.Dataframe) che sono le caratteristiche dei dati ti training 
        labels (pd.Series) che sono le etichette dei dati di training 
        """
        self.features = features
        self.labels = labels 

    def Euclidian_distance(self, point:pd.Series) -> pd.Series: 
        """ 
        con la funzione Euclidian_distances viene calcolata la distanza euclidea tra il punto di test e i campioni di training 

        INPUT: 
        point (pd.Series) che è il punto che si cuole classificare 

        OUTPUT: 
        pd.Series che sono le distanze calcolate 
        """
        return np.sqrt(((self.features - poit)**2).sum(axis=1))
