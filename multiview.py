# # multiview.py
# # Main entry point to run multi-view experiments.
# # Loads data, performs hyperparameter grid search, runs experiments, saves and prints results.

# import os
# import csv
# from utils import (
#     set_seed,
#     grid_search_multiview,
#     run_experiments_multiview,
#     aggregate_metrics,
#     print_aggregated_results
# )
# from data.multiview_data import MultiViewDatasetPreprocessor
# from model.multiview_model.multiview_models import (
#     MultiViewNN_TwoLoss,
#     MultiViewNN_Simul,
#     SimultaneousMultiViewMultiTaskNN,
#     MultiViewMLP
# )

# def main():
#     # Ensure reproducibility
#     set_seed(42)

#     OUTPUT_CSV = "multiview_results.csv"
#     # Remove old results file if it exists
#     if os.path.exists(OUTPUT_CSV):
#         os.remove(OUTPUT_CSV)

#     # Write CSV header
#     with open(OUTPUT_CSV, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Method", "AUC", "Params"])

#     # Load and preprocess data
#     # side_info_path should point to your augmented data CSV
#     dp = MultiViewDatasetPreprocessor(side_info_path='prompting/augmented_data.csv', dataset_id=891)
#     (grid_X, grid_y, grid_Z), (train_X, train_y, train_Z), (test_X, test_y, test_Z) = dp.preprocess()
#     # Define the experiments with model classes and their hyperparameter grids
#     experiments = [
#         {
#             "name": "TwoLoss",
#             "model": MultiViewNN_TwoLoss,
#             "param_grid": {
#                 "hidden": [32, 64],
#                 "lr": [0.001, 0.01],
#                 "epochs": [100],
#                 "lambda_aux": [0.3, 0.5]
#             }
#         },
#         {
#             "name": "Simul",
#             "model": MultiViewNN_Simul,
#             "param_grid": {
#                 "hidden": [32, 64],
#                 "lr": [0.001, 0.01],
#                 "epochs": [100],
#                 "lambda_aux": [0.3, 0.5]
#             }
#         },
#         {
#             "name": "MT-MLP-Simul",
#             "model": SimultaneousMultiViewMultiTaskNN,
#             "param_grid": {
#                 "hidden": [64],
#                 "lr": [0.001, 0.01],
#                 "epochs": [100],
#                 "lambda_aux": [0.3, 0.5],
#                 "lambda_direct": [0.3, 0.5]
#             }
#         },
#         {
#             "name": "MT-MLP",
#             "model": MultiViewMLP,
#             "param_grid": {
#                 "hidden": [32, 64],
#                 "num_layers_x": [3],
#                 "num_layers_z": [2],
#                 "lr": [0.001, 0.01],
#                 "lambda_view": [0.3, 0.5],
#                 "epochs": [100]
#             }
#         }
#     ]

#     # Iterate over each experiment configuration
#     for cfg in experiments:
#         name = cfg["name"]
#         ModelClass = cfg["model"]
#         param_grid = cfg["param_grid"]

#         # Hyperparameter grid search using balanced grid set
#         best_params, best_auc = grid_search_multiview(
#             ModelClass,
#             param_grid,
#             grid_X, grid_Z, grid_y,
#             cv=3
#         )
#         print(f"[{name}] Best params: {best_params}, AUC: {best_auc:.4f}")

#         # Final experiments with best hyperparameters
#         results = run_experiments_multiview(
#             ModelClass,
#             name,
#             num_runs=10,
#             best_params=best_params,
#             train_X=train_X,
#             train_Z=train_Z,
#             train_y=train_y,
#             test_X=test_X,
#             test_Z=test_Z,
#             test_y=test_y
#         )
#         aggregated = aggregate_metrics(results)

#         # Save aggregated results to CSV
#         with open(OUTPUT_CSV, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([name, aggregated.get('auc', ''), str(best_params)])

#         # Print final aggregated results
#         print("\n=== Final Aggregated Results ===")
#         print(f"Method: {name}")
#         print(f"AUC: {aggregated.get('auc', '')}")
#         print(f"Best Parameter Set: {best_params}")
#         print("---------------------------------------------------")
#         print_aggregated_results(name, aggregated, str(best_params))

# if __name__ == "__main__":
#     main()

# multiview.py
import os
import csv
import torch
from utils import (
    set_seed,
    grid_search_multiview,
    run_experiments_multiview,
    aggregate_metrics,
    print_aggregated_results
)
from data.multiview_data import MultiViewDatasetPreprocessor
from model.multiview_model.multiview_models import (
    MultiViewNN_TwoLoss,
    MultiViewNN_Simul,
    SimultaneousMultiViewMultiTaskNN,
    MultiViewMLP
)


def main():
    # Ensure reproducibility
    set_seed(42)

    OUTPUT_CSV = "multiview_results.csv"
    # Remove old results file if it exists
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    # Write CSV header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "AUC", "Params"])

    # Load and preprocess data
    dp = MultiViewDatasetPreprocessor(side_info_path='prompting/augmented_data.csv', dataset_id=891)
    (grid_X, grid_y, grid_Z), (train_X, train_y, train_Z), (test_X, test_y, test_Z) = dp.preprocess()

    # Define experiments
    experiments = [
        # {
        #     "name": "Finetuned and Pretuned MV",
        #     "model": MultiViewNN_TwoLoss,
        #     "param_grid": {
        #         "hidden": [64, 128, 256],
        #         "num_layers_x": [1, 2, 3],
        #         "num_layers_z": [1, 2, 3],
        #         "lr": [0.001, 0.01],
        #         "epochs": [100, 300],
        #         "lambda_aux": [0.3, 0.5]
        #     }
        # }
        # ,
        # {
        #     "name": "MV-Simul",
        #     "model": MultiViewNN_Simul,
        #     "param_grid": {
        #         "hidden": [32, 64],
        #         "lr": [0.001, 0.01],
        #         "epochs": [100],
        #         "lambda_aux": [0.3, 0.5]
        #     }
        # },
        # {
        #     "name": "MT-MV-Simul",
        #     "model": SimultaneousMultiViewMultiTaskNN,
        #     "param_grid": {
        #         "hidden": [64],
        #         "lr": [0.001, 0.01],
        #         "epochs": [100],
        #         "lambda_aux": [0.3, 0.5],
        #         "lambda_direct": [0.3, 0.5]
        #     }
        # },
        {
            "name": "MV MLP",
            "model": MultiViewMLP,
            "param_grid": {
                "hidden": [128, 256],
                "num_layers_x": [1,2,3,4],
                "num_layers_z": [1,2,3,4],
                "lr": [0.001, 0.01],
                "lambda_view": [0.3, 0.5],
                "epochs": [100, 300]
            }
        }
    ]

    # Iterate experiments
    for cfg in experiments:
        name = cfg["name"]
        ModelClass = cfg["model"]
        param_grid = cfg["param_grid"]

        # Hyperparameter grid search
        best_params, best_auc = grid_search_multiview(
            ModelClass,
            param_grid,
            grid_X,
            grid_Z,
            grid_y,
            cv=3
        )
        print(f"[{name}] Best params: {best_params}, AUC: {best_auc:.4f}")

        # Final experiments with best hyperparameters
        results = run_experiments_multiview(
            ModelClass,
            name,
            num_runs=10,
            best_params=best_params,
            train_X=train_X,
            train_Z=train_Z,
            train_y=train_y,
            test_X=test_X,
            test_Z=test_Z,
            test_y=test_y
        )
        print(results, "===")
        aggregated = aggregate_metrics(results)

        # Save aggregated results
        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, aggregated.get('auc', ''), str(best_params)])

        # Print final aggregated results
        print("\n=== Final Aggregated Results ===")
        print(f"Method: {name}")
        print(f"AUC: {aggregated.get('auc', '')}")
        print(f"Best Parameter Set: {best_params}")
        print("---------------------------------------------------")
        print_aggregated_results(name, aggregated, str(best_params))

if __name__ == "__main__":
    main()