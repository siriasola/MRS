import unittest
import pandas as pd
import numpy as np
from validation.evaluation import Evaluation

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        """Inizializza i dati di test. Viene eseguito prima di ogni test per creare un dataset di prova.
          Genera 100 campioni casuali con 5 feature e una variabile target binaria."""
        
        np.random.seed(42) #Fissa la sequenza casuale per ottenere risultati riproducibili.
        self.features = pd.DataFrame(np.random.rand(100, 5))
        self.target = pd.Series(np.random.choice([0, 1], size=100))
        self.k_folds = 5
        self.metriche_scelte = ["Accuracy Rate", "Error Rate", "Sensitivity", "Specificity", "Geometric Mean"]
        self.k = 3
        self.evaluator = Evaluation(self.features, self.target, self.k_folds, self.metriche_scelte, self.k)
    
    def test_valutazione_k_fold(self):

        """Testa la validazione K-Fold. 
        Controlla che i risultati siano un dizionario e che contengano tutte le metriche richieste.
        """
        risultati = self.evaluator.valutazione_k_fold()
        self.assertIsInstance(risultati, tuple)  # Ora gestiamo una tupla
        self.assertIsInstance(risultati[0], dict)  # Il primo elemento deve essere il dizionario delle metriche
        self.assertTrue(all(m in risultati[0] for m in self.metriche_scelte))

    
    def test_valutazione_leave_one_out(self):

        """Testa la validazione Leave-One-Out.
        Controlla che restituisca una tupla e un dizionario con tutte le metriche richieste.
        """

        risultati = self.evaluator.valutazione_leave_one_out()
        self.assertIsInstance(risultati, tuple)
        self.assertIsInstance(risultati[0], dict)
        self.assertTrue(all(m in risultati[0] for m in self.metriche_scelte))

    def test_valutazione_holdout(self):
        
        """Testa la validazione Holdout."""
        risultati = self.evaluator.valutazione_holdout()
        self.assertIsInstance(risultati, tuple)
        self.assertIsInstance(risultati[0], dict)
        self.assertTrue(all(m in risultati[0] for m in self.metriche_scelte))

 
    
if __name__ == '__main__':
    unittest.main()