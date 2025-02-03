import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_confusion_matrix(y_true, y_pred):
    """
    Genera e visualizza una Confusion Matrix senza usare sklearn.
    y_true: array dei valori reali
    y_pred: array dei valori predetti
    """
    # Classi uniche (0 = benigno, 1 = maligno)
    classi = np.unique(y_true)
    
    # Creazione della matrice di confusione
    cm = np.zeros((len(classi), len(classi)), dtype=int)
    
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1

    # Plot della Confusion Matrix
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
    Genera la Curva ROC e calcola l'AUC senza sklearn.
    y_true: array dei valori reali
    y_prob: array delle probabilità di classe 1 (maligno)
    """
    # Ordinare i dati in base alle probabilità predette
    soglie = np.sort(y_prob)[::-1]  # Dal valore più alto al più basso
    tpr = []  # True Positive Rate (Sensibilità)
    fpr = []  # False Positive Rate

    n_positivi = np.sum(y_true == 1)
    n_negativi = np.sum(y_true == 0)

    for soglia in soglie:
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
        df = pd.DataFrame(metriche)
        df.to_csv(filename,index=False)