import pandas as pd
import os
from preprocessing.data_cleaner import DataCleaner, MissingValueHandler, choose_missing_value_method
from preprocessing.normalizzazione import FeatureScaler, user_choose_scaling_method


class DatasetProcessor:
    """
    Classe per il caricamento e preprocessing di un dataset in diversi formati.
    Supporta file CSV, Excel, JSON, TSV e TXT.
    """

    def __init__(self, file_path: str):
        """
        Inizializza il processore del dataset con il percorso del file.
        """
        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[1].lower()  # Estrai l'estensione una sola volta
        self.data = None  # Memorizza il dataset caricato

    def load_data(self):
        """
        Carica il dataset in base all'estensione del file.

        Ritorna:
        pd.DataFrame: Dataset caricato, oppure None in caso di errore.
        """
        try:
            if self.file_extension == ".csv":
                self.data = pd.read_csv(self.file_path)
            elif self.file_extension == ".xlsx":
                self.data = pd.read_excel(self.file_path)
            elif self.file_extension == ".json":
                self.data = pd.read_json(self.file_path)
            elif self.file_extension in (".tsv", ".txt"):
                self.data = pd.read_csv(self.file_path, sep="\t")
            else:
                raise ValueError(f"Formato di file non supportato: {self.file_extension}")

            print("Il dataset è stato caricato correttamente.")
            return self.data

        except Exception as e:
            print(f"Errore nel caricamento del file: {e}")
            return None  # Restituisce None se il caricamento fallisce

def load_and_prepare_data(file_path):
    """Carica, pulisce e pre-processa il dataset."""
    df = DatasetProcessor(file_path).load_data()
    if df is None or df.empty:
        print("Errore: Dataset vuoto o non caricato correttamente.")
        return None, None
    df = DataCleaner("classtype_v1").clean(df)
    df = MissingValueHandler(choose_missing_value_method()).clean(df)
    df = FeatureScaler(target_column="classtype_v1").scale_features(df, user_choose_scaling_method())
        
    features = df.drop(columns=["classtype_v1"], errors='ignore')
    target = df["classtype_v1"]
    return features, target
