
import numpy as np
import pandas as pd

class FeatureScaler:
    """
    Classe per applicare Normalizzazione (Min-Max) o Standardizzazione (Z-score) su un dataset.
    L'utente può scegliere il metodo e la colonna 'classtype_v1' verrà esclusa dalla trasformazione.
    """

    def __init__(self, target_column="classtype_v1"):
        self.target_column = target_column  # Esclude questa colonna dallo scaling

    def scale_features(self, df, method):
        """
        Applica il metodo scelto (Normalizzazione o Standardizzazione) a tutte le colonne numeriche, 
        escludendo la colonna target.

        Parametri:
        df (pd.DataFrame) - Il dataset da scalare
        method (str) - "normalize" per Normalizzazione, "standardize" per Standardizzazione

        Ritorna:
        pd.DataFrame - Dataset scalato
        """
        if method not in ["normalize", "standardize"]:
            print("Il metodo non è valido! Uso di default: Normalizzazione (Min-Max).")
            method = "normalize"

        df_scaled = df.copy()
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != self.target_column]

        if method == "standardize":
            for column in numeric_columns:
                mean_val = df[column].mean()
                std_val = df[column].std()
                df_scaled[column] = (df[column] - mean_val) / std_val
        else:  # method == "normalize"
            for column in numeric_columns:
                min_val = df[column].min()
                max_val = df[column].max()
                if min_val == max_val: # Evita divisione per zero
                    df_scaled[column] = 0
                else:
                    df_scaled[column] = (df[column] - min_val) / (max_val - min_val)

        print(f"\n Metodo applicato a tutte le feature (tranne '{self.target_column}'): {method.upper()}")
        return df_scaled

def user_choose_scaling_method():
    """
    Permette all'utente di scegliere tra Normalizzazione e Standardizzazione.

    Ritorna:
    str - Metodo scelto ("normalize" o "standardize")
    """
    print("\n Scegli il metodo di scaling per tutte le feature:")
    print("1. Normalizzazione (Min-Max Scaling)")
    print("2. Standardizzazione (Z-score Scaling)")

    choice = input("➡️ Inserisci il numero della tua scelta: ").strip()

    if choice == "1":
        return "normalize"
    elif choice == "2":
        return "standardize"
    else:
        print(" Scelta non valida, uso di default: Normalizzazione (Min-Max).")
        return "normalize"