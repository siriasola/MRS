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
        n = len(self.features)
        indices = np.arange(n)
        np.random.shuffle(indices)
        train_end = int(n * train_size)
        
        train_indices = indices[:train_end]
        test_indices = indices[train_end:]
        
        X_train, X_test = self.features.iloc[train_indices], self.features.iloc[test_indices]
        Y_train, Y_test = self.target.iloc[train_indices], self.target.iloc[test_indices]
        
        return X_train, X_test, Y_train, Y_test
   
    def split_k_fold(self):
        n = len(self.features)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // self.k_folds

        X_train_folds, Y_train_folds = [], []
        X_test_folds, Y_test_folds = [], []
        test_indices_folds = [] 

        for i in range(self.k_folds):
            # per l'ultimo fold prendiamo tutti gli indici rimanenti
            if i < self.k_folds - 1:
                test_indices = indices[i * fold_size: (i + 1) * fold_size]
            else:
                test_indices = indices[i * fold_size:]

            train_indices = np.setdiff1d(indices, test_indices)

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
        Divide i dati per Leave-One-Out Cross Validation.
        """
        n = len(self.features) #utilizziamo la lunghezza totale del dataset
        X_train_folds, Y_train_folds, X_test_folds, Y_test_folds = [], [], [], []
        
    #creo un loop per iterare una volta per ogni dato
        for i in range(n):
            test_indices = [i] #test set
            train_indices = np.setdiff1d(np.arange(n), test_indices) #training set è costituito da tutti gli altri dati escluso test_indices
            
            X_train_folds.append(self.features.iloc[train_indices])
            Y_train_folds.append(self.target.iloc[train_indices])
            X_test_folds.append(self.features.iloc[test_indices])
            Y_test_folds.append(self.target.iloc[test_indices])
        
        return X_train_folds, Y_train_folds, X_test_folds, Y_test_folds