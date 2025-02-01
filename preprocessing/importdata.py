import pandas as pd

class DatasetProcessor:
    """
    Classe per il caricamento e preprocessing di un dataset in diversi formati.
    Supporta file CSV, Excel, JSON, TSV e TXT.
    """
    
    def __init__(self, file_path):
        """
        Inizializza il processore del dataset con il percorso del file.
        """
        self.file_path = file_path
        self.data = None  # Il dataset caricato

    def load_data(self):
        """
        Carica il dataset in base all'estensione del file.
        """
        try:
            if self.file_path.endswith('.csv'):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith('.xlsx'):
                self.data = pd.read_excel(self.file_path)
            elif self.file_path.endswith('.json'):
                self.data = pd.read_json(self.file_path)
            elif self.file_path.endswith('.tsv'):
                self.data = pd.read_csv(self.file_path, sep='\t')
            elif self.file_path.endswith('.txt'):
                self.data = pd.read_csv(self.file_path, delimiter='\t')
            else:
                raise ValueError("Formato di file non supportato.")

            print("Il Dataset caricato correttamente.")
        except Exception as e:
            print(f" Errore nel caricamento del file: {e}")
