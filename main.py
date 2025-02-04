import pandas as pd
import os
from preprocessing.importdata import DatasetProcessor
from preprocessing.data_cleaner import DataCleaner, MissingValueHandler, choose_missing_value_method
from preprocessing.normalizzazione import FeatureScaler, user_choose_scaling_method
from validation.datasplit import SplitData
from validation.evaluation import Evaluation
from validation.metriche import MetricheCrossValidation
from validation.visualizzazione import plot_confusion_matrix, plot_roc_curve
from k_nearest_neighbor import ClassificatoreKNN

def load_dataset():
    """Chiede all'utente il percorso del file e carica il dataset."""
    file_path = input("Inserisci il percorso del file del dataset: ").strip()
    if not file_path:
        print("Nessun percorso fornito. Uso di default: 'data/version_1.csv'")
        file_path = 'data/version_1.csv'

    dataset_processor = DatasetProcessor(file_path)
    df = dataset_processor.load_data()

    if df is None or df.empty:
        print("Errore: Il dataset è vuoto o non è stato caricato correttamente.")
        return None
    
    print("\nPrime righe del dataset:")
    print(df.head())
    return df

def clean_dataset(df, target_column="classtype_v1"):
    """Rimuove duplicati e righe con target mancante."""
    cleaner = DataCleaner(target_column)
    df = cleaner.clean(df)
    print("\nDataset pulito:")
    print(df.head())
    return df

def handle_missing_values(df):
    """Permette all'utente di scegliere come riempire i valori mancanti."""
    print("\nCome vuoi gestire i valori mancanti?")
    missing_strategy = choose_missing_value_method()
    print(f"Metodo scelto per i valori mancanti: {missing_strategy.upper()}")

    missing_handler = MissingValueHandler(method=missing_strategy)
    df = missing_handler.clean(df)

    print("\nDati dopo la gestione dei valori mancanti:")
    print(df.head())
    return df

def scale_features(df, target_column="classtype_v1"):
    """Permette all'utente di scegliere e applicare una tecnica di scaling."""
    print("\nCome vuoi scalare le feature?")
    scaling_method = user_choose_scaling_method()
    print(f"Metodo scelto: {scaling_method.upper()}")

    scaler = FeatureScaler(target_column=target_column)
    df_scaled = scaler.scale_features(df, scaling_method)

    print("\nDati dopo il Feature Scaling:")
    print(df_scaled.head())
    return df_scaled

def split_features_target(df, target_column="classtype_v1"):
    """Separa le feature dal target."""
    print("\nSeparazione del dataset in features e target...")
    features = df.drop(columns=[target_column], errors='ignore')
    target = df[target_column]

    print("\nPrime righe delle features:")
    print(features.head())

    print("\nPrime righe del target:")
    print(target.head())

    return features, target

def choose_validation_strategy():
    """Chiede all'utente quale strategia di validazione usare."""
    print("\nScegli la strategia di validazione:")
    print("1. Holdout")
    print("2. K-Fold Cross Validation")
    print("3. Leave-One-Out Cross Validation")
    choice = input("Inserisci la tua scelta (1/2/3): ").strip()

    strategy = None
    test_size = 0.2
    k_folds = 5

    try:
        if choice == '1':  # Holdout
            test_size = float(input("Inserisci la percentuale di test (default 0.2): ").strip() or 0.2)
            strategy = "holdout"
        elif choice == '2':  # K-Fold Cross Validation
            k_folds = int(input("Inserisci il numero di fold (default 5): ").strip() or 5)
            strategy = "k_fold"
        elif choice == '3':  # Leave-One-Out Cross Validation
            strategy = "leave_one_out"
        else:
            print("Scelta non valida. Uso Holdout come default con test_size=0.2.")
            strategy = "holdout"
    except ValueError:
        print("Errore nei parametri. Uso Holdout con test_size=0.2 come default.")
        strategy = "holdout"

    return strategy, test_size, k_folds

def evaluate_model(features, target, strategy, test_size, k_folds):
    """Esegue la validazione e stampa i risultati."""
    metriche_scelte = ["Accuracy Rate", "Error Rate", "Sensitivity", "Specificity", "Geometric Mean"]
    k = int(input("Inserisci il valore di k per il KNN (default 3): ").strip() or 3)

    evaluation = Evaluation(features, target, k_folds=k_folds, metriche_scelte=metriche_scelte, k=k)

    if strategy == "holdout":
        metrics_result = evaluation.valutazione_holdout(train_size=test_size)
    elif strategy == "k_fold":
        metrics_result = evaluation.valutazione_k_fold()
    elif strategy == "leave_one_out":
        metrics_result = evaluation.valutazione_leave_one_out()

    print("\nRisultati della validazione:")
    for metrica, valore in metrics_result.items():
        print(f"{metrica}: {valore:.4f}")

    return target

def visualize_results(target):
    """Genera la matrice di confusione e la curva ROC."""
    print("\nGenerazione della matrice di confusione...")
    plot_confusion_matrix(target.to_numpy(), target.to_numpy())  # Da sostituire con y_pred

    print("\nGenerazione della curva ROC...")
    plot_roc_curve(target.to_numpy(), target.to_numpy())  # Da sostituire con probabilità predette

    print("\nProcesso completato con successo!")

def main():
    df = load_dataset()
    if df is None:
        return
    
    df = clean_dataset(df)
    df = handle_missing_values(df)
    df = scale_features(df)

    features, target = split_features_target(df)
    strategy, test_size, k_folds = choose_validation_strategy()

    target = evaluate_model(features, target, strategy, test_size, k_folds)
    visualize_results(target)

if __name__ == "__main__":
    main()