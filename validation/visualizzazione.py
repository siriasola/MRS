import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred):
    """
    Genera e visualizza una Confusion Matrix.

    y_true: array con le etichette reali
    y_pred: array con le etichette predette
    """
    # Trova le classi uniche nei dati (gestisce numeri e stringhe)
    classi = np.unique(np.concatenate((y_true, y_pred)))
    num_classi = len(classi)

    # Creazione della matrice di confusione vuota
    cm = np.zeros((num_classi, num_classi), dtype=int)

    # Mappare le etichette a indici validi
    class_to_index = {label: idx for idx, label in enumerate(classi)}

    # Riempire la matrice di confusione
    for i in range(len(y_true)):
        true_index = class_to_index[y_true[i]]
        pred_index = class_to_index[y_pred[i]]
        cm[true_index, pred_index] += 1

    # Creazione della figura
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap="Blues")
    plt.colorbar(cax)

    # Etichette sugli assi
    ax.set_xticks(np.arange(num_classi))
    ax.set_yticks(np.arange(num_classi))
    ax.set_xticklabels(classi)
    ax.set_yticklabels(classi)
    plt.xlabel("Predetto")
    plt.ylabel("Reale")

    #Inserire i valori nella matrice
    for i in range(num_classi):
        for j in range(num_classi):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color="black")

    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_true, y_prob):
    """
    Genera la Curva ROC (Receiver Operating Characteristic) e calcola l'AUC (Area Under the Curve) usando 
    il metodo del trapezio)
    y_true: array dei valori reali
    y_prob: array delle probabilità di classe 1 (maligno)
    """
    # Ordinare i dati in base alle probabilità predette e inizializza le liste per i tpr e fpr
    soglie = np.sort(y_prob)[::-1]  # Dal valore più alto al più basso
    tpr = []  # True Positive Rate (Sensibilità)
    fpr = []  # False Positive Rate

    #Conta il numero di esempi positivi e negativi
    n_positivi = np.sum(y_true == 1)
    n_negativi = np.sum(y_true == 0)
    
    
    for soglia in soglie:

        """Per ogni soglia di decisione ( valore di probabilità da usare per distinguere tra classi): 
         genera una previsione binaria e calcola il nuemero di veri positivi e falsi positivi
         registrando i valori di tpr e fpr"""
     
        y_pred = (y_prob >= soglia).astype(int)

        vero_positivo = np.sum((y_pred == 1) & (y_true == 1))
        falso_positivo = np.sum((y_pred == 1) & (y_true == 0))

        tpr.append(vero_positivo / n_positivi)
        fpr.append(falso_positivo / n_negativi)

    # Calcolo dell'AUC con la formula del trapezio
    auc = np.trapz(tpr, fpr)

    # Plot della Curva ROC
    plt.plot(fpr, tpr, marker="o", linestyle="-", label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Linea random
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.show()

    

def salva_risultati_csv(metriche, filename="results/metrics.csv"):
        """ Salva le metriche in un file CSV """
        df = pd.DataFrame(metriche)
        df.to_csv(filename,index=False)