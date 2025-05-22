# File: multitask.py
"""
Main entry point to run multi-task experiments.
Loads data, performs grid search, runs experiments, saves results.
"""
import os
import csv
from model.multitask_model.multitask_models import (
    MultiTaskLogisticRegression,
    MultiTaskNN,
    MultiTaskNN_PretrainFinetuneExtended,
    MultiTaskNN_Decoupled
)
import torch
from data.multitask_data import MultiTaskDatasetPreprocessor
from utils import (
    set_seed,
    grid_search_cv,
    run_experiments,
    aggregate_metrics,
    write_results_csv,
    print_aggregated_results
)

def main():
    # reproducibility
    set_seed(42)

    OUTPUT_CSV = "multitask_results.csv"
    # write CSV header
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "AUC",
            "Acc0", "Prec0", "Rec0", "F10",
            "Acc1", "Prec1", "Rec1", "F11",
            "BCE_Loss", "Params"
        ])

    # preprocess data
    dp = MultiTaskDatasetPreprocessor(
        dataset_id=891,
        side_info_path='prompting/augmented_data.csv'
    )
    (grid_x, grid_y, grid_a1, grid_a2, grid_a3,
     train_x, train_y, ta1, ta2, ta3,
     test_x, test_y) = dp.preprocess()

    # define experiments
    experiments = [
        # {
        #     "name": "MT-LR",
        #     "model": MultiTaskLogisticRegression,
        #     "param_grid": {"lr": [0.001, 0.01]}
        # }
        # {
        #     "name": "MT-MLP",
        #     "model": MultiTaskNN,
        #     "param_grid": {
        #         "lr": [0.001, 0.01],
        #         "hidden_dim": [256],
        #         "num_layers": [1],
        #         "lambda_aux": [0.01]
        #     }
        # }
        # {
        #     "name": "MT-MLP",
        #     "model": MultiTaskNN,
        #     "param_grid": {
        #         "lr": [0.001, 0.01],
        #         "hidden_dim": [64,128,256],
        #         "num_layers": [1,2,3,4],
        #         "lambda_aux": [0.01]
        #     }
        # }
        # ,
        {
            "name": "PretrainFT",
            "model": MultiTaskNN_PretrainFinetuneExtended,
            "param_grid": {
                "lr_pre": [0.01],
                "lr_fine": [0.001],
                "num_layers": [1,2,3,4],
                "hidden_dim":[128],
                "lambda_aux": [0.3],
                "pre_epochs": [300],
                "fine_epochs": [300]
            }
        }
        # {
        #     "name": "Decoupled",
        #     "model": MultiTaskNN_Decoupled,
        #     "param_grid": {
        #         "lr": [0.01],
        #         "hidden_dim":[64, 128, 256],
        #         "num_layers":[1,2],
        #         "lambda_aux": [0.1]
        #     }
        # }
    ]

    # run each experiment
    for cfg in experiments:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        name = cfg["name"]
        Model = cfg["model"]
        grid = cfg["param_grid"]

        # grid search
        best_params, best_auc = grid_search_cv(
            Model, grid,
            grid_x, grid_y,
            cv=3, is_multitask=True,
            aux1=grid_a1, aux2=grid_a2, aux3=grid_a3
        )
        print(f"[{name}] Best params: {best_params}, AUC: {best_auc:.4f}")

        # final runs
        results = run_experiments(
            Model, name, num_runs=10, best_params=best_params,
            train_x_pool=train_x, train_y_pool=train_y,
            test_x_pool=test_x, test_y_pool=test_y,
            is_multitask=True,
            train_aux1_pool=ta1, train_aux2_pool=ta2, train_aux3_pool=ta3
        )
        agg = aggregate_metrics(results)

        # save & print
        write_results_csv(OUTPUT_CSV, name, agg, str(best_params))
        print_aggregated_results(name, agg, str(best_params))


if __name__ == "__main__":
    main()