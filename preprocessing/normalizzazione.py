
import numpy as np
import pandas as pd

class AutoScaler:
    """
    Classe che analizza i dati e sceglie automaticamente se applicare
    Standardizzazione (Z-score) o Normalizzazione (Min-Max).
    """

    def __init__(self):
        self.decisions = {}

    def is_normal_distribution(self, values):
        """
        Determina se i dati seguono una distribuzione normale basandosi sullo skewness (asimmetria).

        Parametri:
        values (np.array) - Serie di valori numerici

        Ritorna:
        bool - True se la distribuzione è normale, False altrimenti
        """
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)

        skewness = (3 * (mean - median)) / std  # Approssimazione della skewness

        # Se il valore assoluto della skewness è < 0.5, la distribuzione è considerata normale
        return abs(skewness) < 0.5

    def choose_scaling_method(self, df):
        """
        Analizza i dati per determinare il miglior metodo di scaling.

        Parametri:
        df (pd.DataFrame) - Il dataset su cui applicare il metodo

        Ritorna:
        pd.DataFrame - Dataset normalizzato
        dict - Metodo scelto per ogni colonna
        str - Metodo finale applicato all'intero dataset
        """
        results = {}

        for column in df.columns:
            values = df[column].dropna().values  # Rimuove i NaN

            # Determina se la distribuzione è normale
            is_normal = self.is_normal_distribution(values)

            # Calcola il range interquartile (IQR) per verificare gli outlier
            Q1, Q3 = np.percentile(values, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            num_outliers = np.sum((values < lower_bound) | (values > upper_bound))

            # Decisione del metodo di scaling
            if is_normal or num_outliers > 0:
                method = "Z-score Standardization"
            else:
                method = "Min-Max Normalization"

            results[column] = method

        # Determina il metodo più usato e applica lo stesso a tutto il dataset
        method_counts = pd.Series(results).value_counts()
        final_method = method_counts.idxmax()

        if final_method == "Z-score Standardization":
            df_scaled = (df - df.mean()) / df.std()
        else:
            df_scaled = (df - df.min()) / (df.max() - df.min())

        self.decisions = results

        return df_scaled, results, final_method