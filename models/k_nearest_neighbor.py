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
        return np.sqrt(((self.features - point)**2).sum(axis=1))

    def k_nearest_neighbor(self, point: pd.Series) -> pd.Series:
        """
        la funzione k_nearest_neighbor trova i vicini più vicini al punto specificato 

        INPUT: 
        point (pd.Series) che è il punto da classificare 

        OUTPUT: 
        tipo p.Series ovvero le etichette dei k più vicini 
        """
        distances = self.Euclidean_distance(point)
        nearest_idx = distances.nsmallest(self.k).index
        return self.labels.loc[nearest_idx]

    def predict(self, point: pd.Series) -> int: 
        """
        la funzione predict ha lo scopo di predire la classe di un punto 

        INPUT: 
        point (pd.Series) che è il punto che vogliamo classificare 

        OUTPUT: 
        restituisce un intero che corrisponde alla classe predetta 
        """
        if self.features is None or self.labels is None: # verifico se il modello è stato addestrato attraverso fit 
            raise ValueError("Il modello non è addestrato! Importante usare fit prima di predict")
        neighbors = self.k_nearest_neighbor(point) # chiamo il metodo k_nearest_neighbor per trovare i k vicini più vicini al mio punto. neighbors sarà del tipo pd.Series e conterrà le labels dei vicini trovati
        c = neighbors.value_counts() # attraverso il metodo value.counts() conto quante volte appare una classe nei vicini 

        # GESTISCO IL CASO DI PAREGGIO 

        if(c == c.max()).sum() >1: # se le classi appaiono lo stesso numero di volte abbiamo il caso di pareggio
            return random.choice(c [c == c.max()].index.tolist()) # nel caso di pareggio si gestisca casualemte
        return c.idxmax()
    
    def predict_batch(self, points: pd.DataFrame) -> pd.Series: 
        """
        Predice le classi per più punti contemporaneamente.

        INPUT:
        points (pd.DataFrame) sono i dataset di punti che vogliamo classificare

        OUTPUT:
        pd.Series che corrispondono alle labels predette per ogni punto
        """
        return points.apply(self.predict, axis=1)

