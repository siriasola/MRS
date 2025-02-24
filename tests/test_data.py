import unittest
import pandas as pd
import numpy as np
from preprocessing.data_cleaner import DataCleaner, MissingValueHandler

class TestDataProcessing(unittest.TestCase):
    """Test per la gestione dei dati, pulizia, duplicati e valori mancanti."""

    def setUp(self):
        self.cleaner = DataCleaner(target_column="Target")  # Corretto "label" in "Target"
        self.missing_handler = MissingValueHandler(method="mean")
        self.data_inconsistent = pd.DataFrame({
            "Feature1": [1, 2, np.nan, 4],  # Rimosso "errore" perché non è un valore numerico
            "Feature2": [np.nan, 2, 3, 4],
            "Target": [1, np.nan, 1, 0]
        })
        self.duplicated_data = pd.DataFrame({
            "feature1": [1, 2, 2, 4, np.nan],
            "feature2": [5, 6, 6, 8, 9],
            "Target": [1, 0, 0, 1, np.nan]  # Corretto "label" in "Target"
        })

    def test_inconsistent_data_handling(self):
        """Testa la pulizia di dati inconsistenti."""
        cleaned_data = self.cleaner.clean(self.data_inconsistent)
        self.assertFalse(cleaned_data.isnull().any().any(), "I valori mancanti o inconsistenti dovrebbero essere rimossi.")

    def test_remove_duplicates(self):
        """Testa la rimozione dei duplicati."""
        cleaned_data = self.cleaner.clean(self.duplicated_data)
        self.assertEqual(len(cleaned_data), 3, "I duplicati dovrebbero essere rimossi dal dataset.")

    def test_remove_missing_target(self):
        """Testa la rimozione di righe con target mancante."""
        cleaned_data = self.cleaner.clean(self.duplicated_data)
        self.assertFalse(cleaned_data['Target'].isna().any(), "Le righe con target mancante dovrebbero essere rimosse.")

    def test_missing_values_filling(self):
        """Testa il riempimento dei valori mancanti."""
        filled_data = self.missing_handler.clean(self.data_inconsistent)
        self.assertFalse(filled_data.isnull().any().any(), "Tutti i valori mancanti dovrebbero essere riempiti.")

if __name__ == "__main__":
    unittest.main()
