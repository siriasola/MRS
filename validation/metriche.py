import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MetricheCrossValidation:
    
    def __init__(self, y_test: pd.Series, previsioni: list, metriche_scelte: list):
        self.y_test = np.array(y_test)
        self.previsioni = np.array(previsioni)
        self.metriche_scelte = metriche_scelte

    def calcolo_metriche(self, y_test, previsioni):
        """
        Calcola le metriche per una singola iterazione.
        """
        vero_positivo = np.sum((y_test == 1) & (previsioni == 1))
        vero_negativo = np.sum((y_test == 0) & (previsioni == 0))
        falso_positivo = np.sum((y_test == 0) & (previsioni == 1))
        falso_negativo = np.sum((y_test == 1) & (previsioni == 0))

        metriche = {}

        if 'Accuracy Rate' in self.metriche_scelte:
            metriche['Accuracy Rate'] = (vero_positivo + vero_negativo) / len(y_test)

        if 'Error Rate' in self.metriche_scelte:
            metriche['Error Rate'] = (falso_positivo + falso_negativo) / len(y_test)

        if 'Sensitivity' in self.metriche_scelte and (vero_positivo + falso_negativo) != 0:
            metriche['Sensitivity'] = vero_positivo / (vero_positivo + falso_negativo)

        if 'Specificity' in self.metriche_scelte and (vero_negativo + falso_positivo) != 0:
            metriche['Specificity'] = vero_negativo / (vero_negativo + falso_positivo)

        if 'Geometric Mean' in self.metriche_scelte and 'Sensitivity' in metriche and 'Specificity' in metriche:
            metriche['Geometric Mean'] = np.sqrt(metriche['Sensitivity'] * metriche['Specificity'])

        return metriche

    def k_fold_cross_validation(self, k):
        """
        Esegue la K-Fold Cross Validation e calcola le metriche per ogni fold.
        """
        n = len(self.y_test)
        fold_size = n // k
        metriche_totali = {m: [] for m in self.metriche_scelte}

        for i in range(k):
            # Definizione degli indici di training e test
            test_indices = list(range(i * fold_size, (i + 1) * fold_size)) if i != k - 1 else list(range(i * fold_size, n))
            train_indices = list(set(range(n)) - set(test_indices))

            y_train, y_test_fold = self.y_test[train_indices], self.y_test[test_indices]
            previsioni_train, previsioni_test = self.previsioni[train_indices], self.previsioni[test_indices]

            # Calcolo metriche per questo fold
            metriche_fold = self.calcolo_metriche(y_test_fold, previsioni_test)
            
            # Salvataggio metriche
            for key in metriche_fold:
                metriche_totali[key].append(metriche_fold[key])

        return metriche_totali

    def leave_one_out_cross_validation(self):
        """
        Esegue la Leave-One-Out Cross Validation e calcola le metriche per ogni iterazione.
        """
        n = len(self.y_test)
        metriche_totali = {m: [] for m in self.metriche_scelte}

        for i in range(n):
            # Test set con un solo elemento
            y_test_loo = np.array([self.y_test[i]])
            previsioni_loo = np.array([self.previsioni[i]])

            # Calcolo metriche per questa iterazione
            metriche_loo = self.calcolo_metriche(y_test_loo, previsioni_loo)

            # Salvataggio metriche
            for key in metriche_loo:
                metriche_totali[key].append(metriche_loo[key])

        return metriche_totali

    def plot_metriche(self, metriche):
        """
        Plotta le metriche ottenute dalla Cross Validation.
        """
        for metrica, valori in metriche.items():
            plt.plot(valori, marker='o', linestyle='solid', linewidth=2, markersize=5, label=metrica)

        plt.xlabel("Iterazioni")
        plt.ylabel("Valore")
        plt.title("Andamento delle metriche nella Cross Validation")
        plt.legend()
        plt.show()
