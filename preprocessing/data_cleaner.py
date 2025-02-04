
import pandas as pd

class DataCleaner:
    """
    Classe per la pulizia del dataset: rimuove duplicati e righe con target mancante.
    """

    def __init__(self, target_column):
        self.target_column = target_column

    def clean(self, df):
        """
        Pulisce il dataset rimuovendo duplicati e righe senza target.

        Parametri:
        df (pd.DataFrame) - Il dataset da pulire

        Ritorna:
        pd.DataFrame - Dataset pulito
        """
        df = df.drop_duplicates()
        df = df.dropna(subset=[self.target_column])  # Elimina righe senza il target
        print("Dataset pulito: duplicati rimossi e target senza valore eliminato.")
        
        df= df.apply(pd.to_numeric, errors='coerce') #stringhe diventano Nan

        print("Valori NaN dopo conversione numerica:\n", df.isna().sum())
        return df


class MissingValueHandler:
    """
    Classe per la gestione dei valori mancanti: permette di scegliere tra media, moda o mediana.
    """

    def __init__(self, method="mean"):
        if method not in ["mean", "median", "mode"]:
            print("Il metodo scelto non valido, uso di default: MEDIA")
            self.method = "mean"
        else:
            self.method = method

    def clean(self, df):
        """
        Riempie i valori mancanti con il metodo scelto.

        Parametri:
        df (pd.DataFrame) - Il dataset da processare

        Ritorna:
        pd.DataFrame - Dataset con valori mancanti riempiti
        """
        for col in df.select_dtypes(include=["number"]).columns:
            if self.method == "mean":
                df[col].fillna(df[col].dropna().mean(), inplace=True)
            elif self.method == "median":
                df[col].fillna(df[col].dropna().median(), inplace=True)
            elif self.method == "mode":
                mode_value = df[col].dropna().mode()
                df[col].fillna(mode_value.iloc[0] if not mode_value.empty else df[col].dropna().median(), inplace=True)

        print(f"Valori mancanti riempiti con {self.method}.")
        return df


def choose_missing_value_method():
    """
    Mostra una lista di opzioni e permette all'utente di scegliere il metodo per riempire i valori mancanti.
    Se l'utente inserisce un valore non valido, viene usata di default la MEDIA.
    """
    print("\nScegli un metodo per riempire i valori mancanti:")
    options = {"1": "mean", "2": "median", "3": "mode"}
    
    choice = input("\nInserisci il numero della tua scelta (1=media; 2=mediana; 3=moda): ").strip()
    return options.get(choice, "mean")  # Default: mean se la scelta non è valida