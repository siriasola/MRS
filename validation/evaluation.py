import numpy as np

class ModelEvaluation:
    def __init__(self, model, features, target):
        self.model = model
        self.features = features
        self.target = target

    def k_fold_cross_validation(self, k):
        n = len(self.features)
        indices = np.arange(n)
        np.random.shuffle(indices)  # Mescolo gli indici per randomizzare la suddivisione
        fold_size = n // k
        scores = []

        for i in range(k):
            test_indices = indices[i * fold_size:(i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)  # Tutti gli altri dati vanno nel training set

            X_train, X_test = self.features.iloc[train_indices], self.features.iloc[test_indices]
            y_train, y_test = self.target.iloc[train_indices], self.target.iloc[test_indices]

            self.model.train(X_train, y_train)  # Addestriamo il modello
            predictions = self.model.predict(X_test)  # Facciamo le predizioni
            score = self.evaluate(y_test, predictions)  # Valutiamo il modello
            scores.append(score)

        return np.mean(scores)  # Restituiamo la media delle performance

    def leave_one_out_cross_validation(self):
        n = len(self.features)
        scores = []

        for i in range(n):
            X_train = self.features.drop(index=i)
            y_train = self.target.drop(index=i)
            X_test = self.features.iloc[i:i+1]  # Prendiamo una sola riga come test
            y_test = self.target.iloc[i:i+1]

            self.model.train(X_train, y_train)  # Addestriamo il modello
            prediction = self.model.predict(X_test)  # Facciamo la predizione
            score = self.evaluate(y_test, prediction)  # Valutiamo il modello
            scores.append(score)

        return np.mean(scores)  # Media delle performance

    def evaluate(self, y_true, y_pred):
        """Calcola una metrica di valutazione (qui errore quadratico medio per regressione)."""
        return np.mean((y_true - y_pred) ** 2)  # Mean Squared Error (MSE)
