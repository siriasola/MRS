import pandas as pd
from preprocessing.importdata import DatasetProcessor
from preprocessing.data_cleaner import DataCleaner, MissingValueHandler, choose_missing_value_method
from preprocessing.normalizzazione import FeatureScaler, user_choose_scaling_method
from validation.evaluation import Evaluation
from validation.visualizzazione import plot_confusion_matrix, plot_metriche_bar, plot_roc_curve
from models import ClassificatoreKNN


def load_and_prepare_data():
    """Carica, pulisce e pre-processa il dataset."""
    file_path = input("Inserisci il percorso del file del dataset: ").strip() or 'data/version_1.csv'
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


def choose_validation_strategy():
    """Permette all'utente di scegliere la strategia di validazione."""
    strategies = {"1": ("holdout", 0.2), "2": ("k_fold", 5), "3": ("leave_one_out", None)}
    choice = input("\nScegli la strategia di validazione:\n1. Holdout\n2. K-Fold Cross Validation\n3. Leave-One-Out Cross Validation\n ").strip()
    return strategies.get(choice, ("holdout", 0.2))


def choose_knn_params():
    """Permette all'utente di scegliere i parametri del KNN."""
    k = int(input("Inserisci il valore di k per il KNN (default 3): ").strip() or 3)
    return k


def evaluate_model(features, target, strategy, param, k):
    """Esegue la valutazione del modello e calcola le metriche."""
    evaluation = Evaluation(features, target, k_folds=(param or 5), metriche_scelte=["Accuracy Rate", "Error Rate", "Sensitivity", "Specificity", "Geometric Mean", "AUC"], k=k)
    
    if strategy == "holdout":
        metrics_result, y_pred, y_test = evaluation.valutazione_holdout(train_size=param)
    elif strategy == "k_fold":
        metrics_result, y_pred = evaluation.valutazione_k_fold()
        y_test = target
    else:  # leave_one_out
        metrics_result, y_pred = evaluation.valutazione_leave_one_out()
        y_test = target
    
    return y_test, y_pred, metrics_result


def visualize_results(y_test, y_pred, metrics_result):
    """Genera visualizzazioni dei risultati e stampa le metriche."""
    print("\n Risultati della validazione:")
    for metrica, valore in metrics_result.items():
        print(f"{metrica}: {valore:.4f}")
    
    plot_confusion_matrix(y_test.to_numpy(), y_pred)
    plot_roc_curve(y_test.to_numpy(), y_pred)
    plot_metriche_bar(metrics_result)
    print("\nProcesso completato con successo!")


def main():
    """Funzione principale del programma."""
    features, target = load_and_prepare_data()
    if features is None or target is None:
        return
    
    strategy, param = choose_validation_strategy()
    k = choose_knn_params()
    y_test, y_pred, metrics_result = evaluate_model(features, target, strategy, param, k)
    visualize_results(y_test, y_pred, metrics_result)


if __name__ == "__main__":
    main()