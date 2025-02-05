import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MetricheCrossValidation:
    
    def __init__(self, metriche_scelte: list):
        """
        Inizializzo la classe con le metriche scelte dall'utente

        INPUT: 
        metriche_scelte (list) che è una lista delle metriche da calcolare
        """
        self.metriche_scelte = metriche_scelte

    def calcolo_metriche(self, y_test: pd.Series, previsioni: pd.Series, probabilita: pd.Series=None) -> dict:
        """
        Calcola le metriche richieste in self.metriche_scelte.
        - y_test: Serie dei valori veri
        - previsioni: Serie dei valori predetti (0/1)
        - probabilita: Serie delle 'probabilità' di classe 1 (se disponibile)
        """


        #print(">>> calcolo_metriche è stato chiamato!")  # Stampa di debug

        #print("Distribuzione y_test:", np.unique(y_test, return_counts=True))
        #print("Distribuzione previsioni:", np.unique(previsioni, return_counts=True))

        """
        Calcola le metriche per una singola iterazione.

        INPUT: 
        y_test (pd.Series) sono i valori reali 
        previsioni (pd.Series) sono i valori predetti
        """

        
        vero_positivo = np.sum((y_test == 1) & (previsioni == 1))
        vero_negativo = np.sum((y_test == 0) & (previsioni == 0))
        falso_positivo = np.sum((y_test == 0) & (previsioni == 1))
        falso_negativo = np.sum((y_test == 1) & (previsioni == 0))

        metriche = {}

        totale = len(y_test) if len(y_test) > 0 else 1  # Evita divisione per zero

        if 'Accuracy Rate' in self.metriche_scelte:
            metriche['Accuracy Rate'] = (vero_positivo + vero_negativo) / totale

        if 'Error Rate' in self.metriche_scelte:
            metriche['Error Rate'] = (falso_positivo + falso_negativo) / totale

        if 'Sensitivity' in self.metriche_scelte:
            denominatore_sens = (vero_positivo + falso_negativo)
            metriche['Sensitivity'] = vero_positivo / denominatore_sens if denominatore_sens != 0 else 0.0

        if 'Specificity' in self.metriche_scelte:
            denominatore_spec = (vero_negativo + falso_positivo)
            metriche['Specificity'] = vero_negativo / denominatore_spec if denominatore_spec != 0 else 0.0

        if 'Geometric Mean' in self.metriche_scelte:
            sensitivity = metriche.get('Sensitivity', vero_positivo / (vero_positivo + falso_negativo) if (vero_positivo + falso_negativo) != 0 else 0.0)
            specificity = metriche.get('Specificity', vero_negativo / (vero_negativo + falso_positivo) if (vero_negativo + falso_positivo) != 0 else 0.0)
            
            metriche['Geometric Mean'] = np.sqrt(sensitivity * specificity)

        if 'AUC' in self.metriche_scelte:
            if probabilita is None:
                # Se non abbiamo la probabilità, non possiamo calcolare l'AUC.
                metriche['AUC'] = None
            else:
                metriche['AUC'] = self.calcolo_auc(y_test, probabilita)

        return metriche
    
    def calcolo_auc(self, y_true: pd.Series, y_prob: pd.Series) -> float:
        """
        Calcola l'Area Under the ROC Curve (AUC) basandosi sulle probabilità di classe 1.
        """
        # Ordiniamo le soglie dal più alto al più basso
        soglie = np.sort(y_prob)[::-1]
        tpr = []
        fpr = []
        
        n_positivi = np.sum(y_true == 1)
        n_negativi = np.sum(y_true == 0)

        # Per ogni soglia, costruiamo una predizione binaria e calcoliamo TPR e FPR
        for soglia in soglie:
            y_pred_temp = (y_prob >= soglia).astype(int)
            vp = np.sum((y_pred_temp == 1) & (y_true == 1))
            fp = np.sum((y_pred_temp == 1) & (y_true == 0))

            tpr.append(vp / n_positivi if n_positivi > 0 else 0.0)
            fpr.append(fp / n_negativi if n_negativi > 0 else 0.0)

        # Calcolo dell'area con la formula del trapezio
        auc_value = np.trapz(tpr, fpr)
        return auc_value


    def holdout_validation_metrics(self,model, X_train, X_test, Y_train, Y_test) -> dict:
        """
        Esegue la validazione Holdout e calcola le metriche.

        Args:
        X_train (pd.DataFrame): Dati di training.
        X_test (pd.DataFrame): Dati di test.
        Y_train (pd.Series): Target di training.
        Y_test (pd.Series): Target di test.

        Returns:
        tipo dict ce sono le metriche calcolate.
        """
        model.train(X_train, Y_train)
        previsioni = model.predict(X_test)
        metriche = self.calcolo_metriche(Y_test, previsioni)
        return metriche
    

    def k_fold_cross_validation(self, model, splits:tuple) ->dict:
        """
        Esegue la K-Fold Cross Validation e calcola le metriche per ogni fold.

        INPUT: 
        model che è il modello di apprendimento automatico
        splits (tuple) che sono i dati divisi in training e test set per ogni fold

        OUTPUT: 
        tipo dict che sono le metriche aggregate 
        """
        X_train_folds, Y_train_folds, X_test_folds, Y_test_folds = splits
        metriche_totali = {m: [] for m in self.metriche_scelte}

        for i in range(len(X_train_folds)):
            model.train(X_train_folds[i], Y_train_folds[i])
            previsioni = model.predict(X_test_folds[i])
            # Calcolo metriche per questo fold
            metriche_fold = self.calcolo_metriche(Y_test_folds[i], previsioni)
            
            # Salvataggio metriche
            for key, value in metriche_fold.items():
                metriche_totali[key].append(value)

        return {key: np.mean(values) for key, values in metriche_totali.items()}

    def leave_one_out_cross_validation(self, model, splits:tuple) -> dict:
        """
        Esegue la Leave-One-Out Cross Validation e calcola le metriche per ogni iterazione.

        INPUT: 
        model che è il modello di apprendimento automatico 
        splits (tuple) ovvero i dati divisi per ogni iterazione 

        OUTPUT: 
        tipo dict che sono le metriche aggregate
        """
        X_train_folds, Y_train_folds, X_test_folds, Y_test_folds = splits

        metriche_totali = {m: [] for m in self.metriche_scelte}

        for i in range(len(X_train_folds)):
            model.train(X_train_folds[i], Y_train_folds[i])
            previsioni = model.predict(X_test_folds[i])

            # Calcolo metriche per questa iterazione
            metriche_loo = self.calcolo_metriche(Y_test_folds[i], previsioni)

            # Salvataggio metriche
            for key, value in metriche_loo.items():
                metriche_totali[key].append(value)

        return {key: np.mean(values) for key, values in metriche_totali.items()}

    def plot_metriche(self, metriche: dict) -> None:
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

    def salva_metriche_csv(self, metriche, filename="results/metrics.csv"):
        import os 
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        df = pd.DataFrame([metriche])
