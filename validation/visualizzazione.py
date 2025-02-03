import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_confusion_matrix(y_true, y_pred):
    """
    Genera e visualizza una Confusion Matrix. 
    Mostra il numero di veri positivi, veri negativi, falsi positivi e falsi negativi
    y_true: array dei valori reali
    y_pred: array dei valori predetti
    """
    # Classi uniche (0 = benigno, 1 = maligno): trova le classi uniche nei dati di input
    classi = np.unique(y_true)
    
    # Creazione della matrice di confusione con valori inizializzati a zero
    cm = np.zeros((len(classi), len(classi)), dtype=int)
    
    for i in range(len(y_true)):

        """Riempie la matrice di confusione contando quante volte il modello ha previsto correttamente
         e scorrettamente ciascuna classe"""
        
        cm[y_true[i], y_pred[i]] += 1

    # Plot della Confusion Matrix

    """Visualizzazione della matrice tramite matplotlib: 
     usa matshow per rappresentarla con i colori e agigunge le etichette sugli assi per indicare
      "Benigno" (0) e "Maligno" (1) ed infine inserisce i valori numerici direttamente nella matrice"""
    
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap="Blues")
    plt.colorbar(cax)

    # Etichette
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benigno", "Maligno"])
    ax.set_yticklabels(["Benigno", "Maligno"])
    plt.xlabel("Predetto")
    plt.ylabel("Reale")

    # Mostra i numeri dentro la matrice
    for i in range(len(classi)):
        for j in range(len(classi)):
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