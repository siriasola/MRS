import unittest
import pandas as pd
from models.k_nearest_neighbor import ClassificatoreKNN

class TestKNNClassifier(unittest.TestCase):
    """Test per il classificatore KNN, inclusi addestramento, errori e predizioni."""

    def setUp(self):
        self.knn = ClassificatoreKNN(k=3)

    def test_knn_untrained_error(self):
        """Verifica che venga sollevato un errore se il modello KNN non è addestrato."""
        with self.assertRaises(ValueError):
            self.knn.predict(pd.Series([1, 2]))

    def test_knn_prediction(self):
        """Testa una predizione KNN con un piccolo dataset di test."""
        self.knn.train(pd.DataFrame({"Feature1": [1, 2, 3], "Feature2": [2, 3, 4]}), pd.Series([0, 1, 0]))
        prediction = self.knn.predict(pd.Series([1.5, 2.5]))
        self.assertIn(prediction, [0, 1], "La predizione dovrebbe essere una delle classi previste.")

if __name__ == "__main__":
    unittest.main()
