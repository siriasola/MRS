import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from validation.datasplit import SplitData
from models.k_nearest_neighbor import ClassificatoreKNN
from validation.metriche import MetricheCrossValidation

class Evaluation:
    def __init__(self, features: pd.DataFrame, target: pd.Series, k_folds: int, metriche_scelte: list, k: int):
        """
        Inizializza la classe Evaluation per valutare il modello con tecniche di validazione incrociata.
        
        :param features: DataFrame contenente le feature di input.
        :param target: Serie contenente le etichette di classe.
        :param k_folds: Numero di fold per la validazione incrociata.
        :param metriche_scelte: Lista delle metriche da calcolare.
        :param k: Numero di vicini da considerare nel KNN.
        """
        self.features = features
        self.target = target
        self.k_folds = k_folds
        self.metriche_scelte = metriche_scelte
        self.k = k
        
        # Creazione di un'istanza della classe SplitData per suddividere i dati
        self.Split = SplitData(features, target, k_folds)

    def valutazione_k_fold(self):
        """
        Esegue la validazione incrociata K-Fold e calcola le metriche per ciascun fold.
        """
        X_train_folds, Y_train_folds, X_test_folds, Y_test_folds = self.Split.split_k_fold()
        metriche_totali = {m: [] for m in self.metriche_scelte}
        
        for i in range(self.k_folds):
            modello_knn = ClassificatoreKNN(self.k)
            modello_knn.train(X_train_folds[i].to_numpy(), Y_train_folds[i].to_numpy())
            previsioni = modello_knn.predict(X_test_folds[i].to_numpy())
            
            C_Metriche = MetricheCrossValidation(self.metriche_scelte)
            metriche_fold = C_Metriche.calcolo_metriche(Y_test_folds[i].to_numpy(), previsioni)
            
            for key, value in metriche_fold.items():
                metriche_totali[key].append(value)
        
        return {key: np.mean(values) for key, values in metriche_totali.items()}
    
    def valutazione_leave_one_out(self):
        """
        Esegue la validazione Leave-One-Out e calcola le metriche per ogni iterazione.
        """
        X_train_folds, Y_train_folds, X_test_folds, Y_test_folds = self.Split.split_leave_one_out()
        metriche_totali = {m: [] for m in self.metriche_scelte}
        
        for i in range(len(self.features)):
            modello_knn = ClassificatoreKNN(self.k)
            modello_knn.train(X_train_folds[i].to_numpy(), Y_train_folds[i].to_numpy())
            previsioni = modello_knn.predict(X_test_folds[i].to_numpy())
            
            C_Metriche = MetricheCrossValidation(self.metriche_scelte)
            metriche_loo = C_Metriche.calcolo_metriche(Y_test_folds[i].to_numpy(), previsioni)
            
            for key, value in metriche_loo.items():
                metriche_totali[key].append(value)
        
        return {key: np.mean(values) for key, values in metriche_totali.items()}
    
    def valutazione_holdout(self, train_size=0.8):
        """
        Esegue la validazione Holdout e calcola le metriche.
        
        :param train_size: Percentuale di dati da usare per il training (default: 80%)
        """
        X_train, X_test, Y_train, Y_test = self.Split.split_holdout(train_size)
        
        modello_knn = ClassificatoreKNN(self.k)
        modello_knn.train(X_train.to_numpy(), Y_train.to_numpy())
        previsioni = modello_knn.predict(X_test.to_numpy())
        
        C_Metriche = MetricheCrossValidation(self.metriche_scelte)
        return C_Metriche.calcolo_metriche(Y_test.to_numpy(), previsioni)
