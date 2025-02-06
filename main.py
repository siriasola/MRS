import pandas as pd
import os
from preprocessing.importdata import DatasetProcessor, load_and_prepare_data
from preprocessing.data_cleaner import DataCleaner, MissingValueHandler, choose_missing_value_method
from preprocessing.normalizzazione import FeatureScaler, user_choose_scaling_method
from validation.evaluation import Evaluation, evaluate_model
from validation.visualizzazione import plot_confusion_matrix, plot_metriche_bar, plot_roc_curve
from models import ClassificatoreKNN
from save_results_to_excel import save_results_to_excel

def get_user_inputs():
    """Chiede tutti gli input all'utente in una singola funzione."""
    file_path = input("Inserisci il percorso del file del dataset: ").strip() or 'data/version_1.csv'

    print("\nScegli la strategia di validazione:")
    print("1. Holdout\n2. K-Fold Cross Validation\n3. Leave-One-Out Cross Validation")
    strategies = {"1": ("holdout", 0.2), "2": ("k_fold", 5), "3": ("leave_one_out", None)}
    strategy, param = strategies.get(input("Scelta: ").strip(), ("holdout", 0.2))

    k = int(input("Inserisci il valore di k per il KNN (default 3): ").strip() or 3)

    available_metrics = ["Accuracy Rate", "Error Rate", "Sensitivity", "Specificity", "Geometric Mean", "AUC"]
    print("\nScegli le metriche da calcolare:")
    for i, metric in enumerate(available_metrics, 1):
        print(f"{i}. {metric}")
    print(f"{len(available_metrics) + 1}. All the above")
    metric_choice = input("Scelta: ").strip()

    if metric_choice.lower() == 'all' or str(len(available_metrics) + 1) in metric_choice.split(","):
        metrics = available_metrics
    else:
        selected_indices = {int(ch.strip()) for ch in metric_choice.split(",") if ch.strip().isdigit()}
        metrics = [available_metrics[i - 1] for i in selected_indices if 1 <= i <= len(available_metrics)]

    return file_path, strategy, param, k, metrics


def main():
    """Funzione principale del programma."""
    file_path, strategy, param, k, metrics = get_user_inputs()

    features, target = load_and_prepare_data(file_path)
    if features is None or target is None:
        return

    y_test, y_pred, metrics_result = evaluate_model(features, target, strategy, param, k, metrics)
    save_results_to_excel(metrics_result, y_test, y_pred)
  
if __name__ == "__main__":
    main()
