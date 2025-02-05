import numpy as np
import pandas as pd

class FeatureScaler:
    """
    Classe per applicare Normalizzazione (Min-Max) o Standardizzazione (Z-score) su un dataset.
    Esclude le colonne 'classtype_v1' (target) e 'Sample code number' (identificativo) dallo scaling.
    """

    def __init__(self, exclude_columns=None):
        """
        Inizializza il FeatureScaler con le colonne da escludere.

        Parametri:
        exclude_columns (list) - Lista di colonne da escludere dallo scaling. Default: ['classtype_v1', 'Sample code number']
        """
        if exclude_columns is None:
            exclude_columns = ["classtype_v1", "Sample code number"]
        self.exclude_columns = exclude_columns

    def scale_features(self, df, method):
        """
        Applica il metodo scelto (Normalizzazione o Standardizzazione) a tutte le colonne numeriche, 
        escludendo le colonne specificate.

        Parametri:
        df (pd.DataFrame) - Il dataset da scalare
        method (str) - "normalize" per Normalizzazione, "standardize" per Standardizzazione

        Ritorna:
        pd.DataFrame - Dataset scalato
        """
        if method not in ["normalize", "standardize"]:
            print("Metodo non valido! Uso di default: Normalizzazione (Min-Max).")
            method = "normalize"

        df_scaled = df.copy()
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in self.exclude_columns]

        if method == "standardize":
            for column in numeric_columns:
                mean_val = df[column].mean()
                std_val = df[column].std()
                df_scaled[column] = (df[column] - mean_val) / std_val
        else:  # method == "normalize"
            for column in numeric_columns:
                min_val = df[column].min()
                max_val = df[column].max()
                df_scaled[column] = (df[column] - min_val) / (max_val - min_val)

        print(f"\nMetodo applicato a tutte le feature (tranne {self.exclude_columns}): {method.upper()}")
        return df_scaled

class FeatureScalerStrategyManager:
    """
    Classe per la gestione dello scaling delle feature in base alla strategia scelta.
    """

    def __init__(self, exclude_columns=None):
        """
        Inizializza la gestione dello scaling con le colonne da escludere.

        Parametri:
        exclude_columns (list) - Lista di colonne da escludere dallo scaling.
        """
        self.exclude_columns = exclude_columns if exclude_columns else ["classtype_v1", "Sample code number"]

    def scale_features(self, strategy: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gestisce lo scaling delle feature utilizzando la strategia specificata.

        Parametri:
        strategy (str) - "normalize" per Normalizzazione o "standardize" per Standardizzazione.
        data (pd.DataFrame) - Il dataset da scalare.

        Ritorna:
        pd.DataFrame - Dataset con scaling applicato.
        """
        scaler = FeatureScaler(self.exclude_columns)
        return scaler.scale_features(data, strategy)

def user_choose_scaling_method():
    """
    Permette all'utente di scegliere tra Normalizzazione e Standardizzazione.

    Ritorna:
    str - Metodo scelto ("normalize" o "standardize")
    """
    print("\nScegli il metodo di scaling per tutte le feature:")
    print("1. Normalizzazione (Min-Max Scaling)")
    print("2. Standardizzazione (Z-score Scaling)")

    choice = input("➡️ Inserisci il numero della tua scelta: ").strip()

    if choice == "1":
        return "normalize"
    elif choice == "2":
        return "standardize"
    else:
        print("Scelta non valida, uso di default: Normalizzazione (Min-Max).")
        return "normalize"