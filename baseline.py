# baseline_all.py
import os
import csv
from utils import set_seed, grid_search_cv_basline, run_experiments_baseline, aggregate_metrics, write_results_csv, print_aggregated_results
from data.baseline_data import BaselineDatasetPreprocessor
from model.baseline_model import BaselineMLP, BaselineLogisticRegression, XGBoostModel, RandomForestModel
import torch

# Ensure reproducibility
def main():
    set_seed(42)

    # When first generating the file
    # OUTPUT_CSV = "final_results.csv"
    # # # Remove old results file if exists
    # # if os.path.exists(OUTPUT_CSV):
    # #     os.remove(OUTPUT_CSV)
    # # Write CSV header
    # with open(OUTPUT_CSV, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #         "Method", "AUC", "Overall_Acc",
    #         "Acc_Class0", "Prec_Class0", "Rec_Class0", "F1_Class0",
    #         "Acc_Class1", "Prec_Class1", "Rec_Class1", "F1_Class1",
    #         "BCE_Loss", "Params"
    #     ])

    # Data loading and preprocessing
    dp = BaselineDatasetPreprocessor(dataset_id=891)
    grid_x, grid_y, pool_x, pool_y, test_x, test_y = dp.preprocess()

    # Experiment definitions for each model
    experiments = [
        {
            "name": "BaselineMLP",
            "model": BaselineMLP,
            "param_grid": {"lr": [0.001, 0.01], "hidden_dim": [64, 128, 256], "num_layers": [1, 2, 3, 4]}
        },
        {
            "name": "LogisticRegression",
            "model": BaselineLogisticRegression,
            "param_grid": {"lr": [0.001, 0.01]}
        },
        {
            "name": "XGBoost",
            "model": XGBoostModel,
            "param_grid": {"learning_rate": [0.001, 0.01], "max_depth": [3,5,7], "n_estimators": [100, 300]}
        },
        {
            "name": "RandomForest",
            "model": RandomForestModel,
            "param_grid": {"max_depth": [3,5,7], "n_estimators": [100, 300]}
        }
    ]

    # Loop over each experimental setup
    for cfg in experiments:
        if torch.cuda.is_available():
            print("True")
            torch.cuda.empty_cache()
        name = cfg["name"]
        ModelClass = cfg["model"]
        param_grid = cfg["param_grid"]

        # Hyperparameter grid search
        best_params, best_auc = grid_search_cv_basline(
            ModelClass, param_grid, grid_x, grid_y,
            cv=3
        )
        print(f"[{name}] Best params: {best_params}, AUC: {best_auc:.4f}")

        # Final experiments and metric aggregation
        results = run_experiments_baseline(
            ModelClass, name, num_runs=10,
            best_params=best_params,
            train_x_pool=pool_x, train_y_pool=pool_y,
            test_x_pool=test_x, test_y_pool=test_y
        )
        aggregated = aggregate_metrics(results)

        # # Save and print results
        # write_results_csv(OUTPUT_CSV, name, aggregated, str(best_params))
        print_aggregated_results(name, aggregated, str(best_params))

if __name__ == "__main__":
    main()

