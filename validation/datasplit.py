import numpy as np
import pandas as pd

class SplitData:

    """Implementa tre tecniche per la suddivisione dei dati:
    - Holdout
    - K-Fold Cross Validation
    - Leave-One-Out Cross Validation
    """
    
    def __init__(self, features: pd.DataFrame, target: pd.Series, k_folds: int):
        self.features = features  # DataFrame contenente le caratteristiche
        self.target = target  # Serie con i valori target
        self.k_folds = k_folds  # Numero di fold per la K-Fold Cross Validation


    def split_holdout(self, train_size=0.8):
        """
        Divide i dati in training e test set secondo la proporzione specificata.
        """
        n = len(self.features) #dimensione totale del dataset
        indices = np.arange(n) #crea array di indici da 0 a n-1
        np.random.shuffle(indices) #mescola casualmente gli indici per evitare bias nella divisione
        train_end = int(n * train_size) #calcola il numero di elementi da assegnare al training set in base al trai_size 
        
        train_indices = indices[:train_end] #contiene i primi indici per il training set 
        test_indices = indices[train_end:] #contiene gli indici rimanenti per il test set
        
        """Suddivisione dellle features e target in base agli indici generati"""

        X_train, X_test = self.features.iloc[train_indices], self.features.iloc[test_indices]
        Y_train, Y_test = self.target.iloc[train_indices], self.target.iloc[test_indices]
        
        return X_train, X_test, Y_train, Y_test
   
    def split_k_fold(self):
        n = len(self.features) #dimensione totale del dataset
        indices = np.arange(n) #crea array di indici da 0 a n-1
        np.random.shuffle(indices) #mescola casualmente gli indici 
        fold_size = n // self.k_folds #calcola la dimensione di ogni fold (divisione intera)

        """Inizializziamo le liste per salvare i folds (insiemi di training e test)"""

        X_train_folds, Y_train_folds = [], []
        X_test_folds, Y_test_folds = [], []
        test_indices_folds = [] 

        """Generiamo un loop per creare i folds"""

        for i in range(self.k_folds):

            """Se non è l'ultimo fold, assegna fold_size elementi al test set, altrimenti assegna i restanti elementi"""

            if i < self.k_folds - 1:
                test_indices = indices[i * fold_size: (i + 1) * fold_size]
            else:
                test_indices = indices[i * fold_size:]

            train_indices = np.setdiff1d(indices, test_indices) #gli indici di training sono tutti gli indici esclusi quelli di test

            # Salvataggio dei vari set
            X_train_folds.append(self.features.iloc[train_indices])
            Y_train_folds.append(self.target.iloc[train_indices])
            X_test_folds.append(self.features.iloc[test_indices])
            Y_test_folds.append(self.target.iloc[test_indices])

            # Salviamo anche gli indici di test
            test_indices_folds.append(test_indices)

        return X_train_folds, Y_train_folds, X_test_folds, Y_test_folds, test_indices_folds

    #definiamo il metodo per effetturale la Leave-One-Out Cross Validation
    def split_leave_one_out(self):
        """
        Divide i dati per Leave-One-Out Cross Validation, dove ogni campione viene usato come test una volta sola.

        """
        n = len(self.features) #utilizziamo la lunghezza totale del dataset
        X_train_folds, Y_train_folds, X_test_folds, Y_test_folds = [], [], [], []
        
    #creo un loop per iterare una volta per ogni dato
        for i in range(n):
            test_indices = [i] #test set di cui seleziona un solo dato
            train_indices = np.setdiff1d(np.arange(n), test_indices) #training set è costituito da tutti gli altri dati escluso test_indices
            
            X_train_folds.append(self.features.iloc[train_indices])
            Y_train_folds.append(self.target.iloc[train_indices])
            X_test_folds.append(self.features.iloc[test_indices])
            Y_test_folds.append(self.target.iloc[test_indices])
        
        return X_train_folds, Y_train_folds, X_test_folds, Y_test_folds