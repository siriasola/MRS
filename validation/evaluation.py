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
        """
        self.features = features.apply(pd.to_numeric, errors='coerce')  # Converte le feature in numeri
        self.features.fillna(self.features.mean(), inplace=True)  # Sostituisce valori mancanti con la media

        self.target = target
        self.k_folds = k_folds
        self.metriche_scelte = metriche_scelte
        self.k = k

        # Creazione di un'istanza della classe SplitData per suddividere i dati
        self.Split = SplitData(features, target, k_folds)



    def valutazione_k_fold(self):
    # Ora la split_k_fold restituisce 5 valori
        X_train_folds, Y_train_folds, X_test_folds, Y_test_folds, test_indices_folds = self.Split.split_k_fold()

        metriche_totali = {m: [] for m in self.metriche_scelte}
        
        # Inizializziamo un array vuoto (o np.zeros) per TUTTI i campioni = len(self.target)
        # Ciò ci consentirà di inserire le predizioni al posto giusto
        y_pred_all = np.zeros(len(self.target), dtype=int)

        for i in range(self.k_folds):
            # Istanzia e addestra il modello su X_train_folds[i]
            modello_knn = ClassificatoreKNN(self.k)
            modello_knn.train(X_train_folds[i], Y_train_folds[i])

            # Calcola le predizioni sul test fold
            previsioni = modello_knn.predict_batch(X_test_folds[i])
            probabilita = modello_knn.predict_proba_batch(X_test_folds[i])

            # Assegniamo le predizioni negli indici corrispondenti
            y_pred_all[test_indices_folds[i]] = previsioni
            
            # Calcola le metriche per questo fold
            C_Metriche = MetricheCrossValidation(self.metriche_scelte)
            metriche_fold = C_Metriche.calcolo_metriche(Y_test_folds[i], previsioni, probabilita)

            # Aggiungi i risultati di fold in fold
            for key, value in metriche_fold.items():
                metriche_totali[key].append(value)

        # Calcoliamo la media delle metriche su tutti i fold
        metriche_medie = {key: np.mean(values) for key, values in metriche_totali.items()}
        return metriche_medie, y_pred_all  # y_pred_all è ordinato correttamente!

    def valutazione_leave_one_out(self):
        """
        Esegue la validazione Leave-One-Out e calcola le metriche per ogni iterazione.
        """
        X_train_folds, Y_train_folds, X_test_folds, Y_test_folds = self.Split.split_leave_one_out()
        metriche_totali = {m: [] for m in self.metriche_scelte}
        y_pred_totale = []

        for i in range(len(self.features)):
            modello_knn = ClassificatoreKNN(self.k)

            X_train = pd.DataFrame(X_train_folds[i])
            Y_train = pd.Series(Y_train_folds[i])

            modello_knn.train(X_train, Y_train)
            previsioni = modello_knn.predict_batch(pd.DataFrame(X_test_folds[i])) 
            
            probabilita = modello_knn.predict_proba_batch(pd.DataFrame(X_test_folds[i])) 
            
            y_pred_totale.append(previsioni.iloc[0])  # Un solo elemento per iterazione

            C_Metriche = MetricheCrossValidation(self.metriche_scelte)
            metriche_loo = C_Metriche.calcolo_metriche(pd.Series(Y_test_folds[i]), previsioni, probabilita)

            for key, value in metriche_loo.items():
                metriche_totali[key].append(value)

        metriche_medie = {key: np.mean(values) for key, values in metriche_totali.items()}
        return metriche_medie, np.array(y_pred_totale)  # Ora restituisce anche le previsioni!

    def valutazione_holdout(self, train_size=0.8):
        """
        Esegue la validazione Holdout e calcola le metriche.
        """
        X_train, X_test, Y_train, Y_test = self.Split.split_holdout(train_size)

        X_train = pd.DataFrame(X_train)
        Y_train = pd.Series(Y_train)
        X_test = pd.DataFrame(X_test)
        Y_test = pd.Series(Y_test)

        modello_knn = ClassificatoreKNN(self.k)
        modello_knn.train(X_train, Y_train)
        previsioni = modello_knn.predict_batch(X_test)

        # Probabilità [0..1] della classe 1 (per AUC)
        probabilita = modello_knn.predict_proba_batch(X_test)

        C_Metriche = MetricheCrossValidation(self.metriche_scelte)
        metriche_holdout = C_Metriche.calcolo_metriche(Y_test, previsioni, probabilita) #aggiunta di probabilita

        return metriche_holdout, np.array(previsioni), Y_test  # Ora restituisce anche le previsioni!
