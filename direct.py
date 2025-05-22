import os
import csv
from utils import (
    set_seed,
    grid_search_direct,
    run_experiments_direct,
    aggregate_metrics,
    write_results_csv,
    print_aggregated_results,
)
import torch
from data.direct_data import DirectDataPreprocessor
from model.direct_model.direct_models import DirectPatternNoDecomp, DirectDecoupledResidualNoDecomp, DirectDecoupledResidualModel, DirectPatternResidual

def main():
    # Ensure reproducibility
    set_seed(42)

    OUTPUT_CSV = "final_results.csv"

    # # Write CSV header
    # with open(OUTPUT_CSV, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #         "Method", "AUC", "Overall_Acc",
    #         "Acc_Class0", "Prec_Class0", "Rec_Class0", "F1_Class0",
    #         "Acc_Class1", "Prec_Class1", "Rec_Class1", "F1_Class1",
    #         "BCE_Loss", "Params"
    #     ])

    # Load and preprocess data
    dp = DirectDataPreprocessor(dataset_id=891)
    grid_x, grid_y, grid_z, train_x, train_y, train_z, test_x, test_y = dp.preprocess()

    # Define experiments for each direct model
    experiments = [
        {
            "name": "Simul_NoDecomp",
            "model": DirectPatternNoDecomp,
            # "param_grid": {
            #     "hidden_dim": [32, 64, 128, 256],
            #     "num_layers": [1, 2, 3, 4],
            #     "residual_type": ["concat", "sum"],
            #     "lr": [0.001, 0.01]
            # }
            "param_grid": {
                "hidden_dim": [32, 64, 128, 256],
                "num_layers": [1, 2, 3, 4],
                "residual_type": ["concat", "sum"],
                "lr": [0.001, 0.01]
            }
        }
        ,
        {
            "name": "Simul_with_Decomp",
            "model": DirectPatternResidual,
            # "param_grid": {
            #     "hidden_dim": [32, 64, 128, 256],
            #     "num_layers": [1, 2, 3, 4]
            #     "lr": [0.001, 0.01]
            # }
            "param_grid": {
                "hidden_dim": [32, 64, 128, 256],
                "num_layers": [1, 2],
                "lr": [0.001, 0.01]
            }
        }
        ,
        {
            "name": "Decoupled_NoDecomp",
            "model": DirectDecoupledResidualNoDecomp,
            # "param_grid": {
            #     "hidden_dim": [32, 64, 128, 256],
            #     "num_layers": [1, 2, 3, 4],
            #     "residual_type": ["concat", "sum"],
            #     "lr": [0.001, 0.01]
            # }
            "param_grid": {
                "hidden_dim": [32, 64, 128],
                "num_layers": [1, 2],
                "lr": [0.001]
            }
        }
        ,
        {
            "name": "Decoupled_with_Decomp",
            "model": DirectDecoupledResidualModel,
            # "param_grid": {
            #     "hidden_dim": [32, 64, 128, 256],
            #     "num_layers": [1, 2, 3, 4],
            #     "residual_type": ["concat", "sum"],
            #     "lr": [0.001, 0.01]
            # }
            "param_grid": {
                "hidden_dim": [64],
                "num_layers": [1],
                "lr": [0.01]
            }
        }
    ]

    # Run experiments for each configuration
    for cfg in experiments:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        name = cfg["name"]
        ModelClass = cfg["model"]
        param_grid = cfg["param_grid"]

        # Hyperparameter grid search
        best_params, best_auc = grid_search_direct(
            ModelClass,
            param_grid,
            grid_x, grid_y, grid_z,
            cv=3,
            early_stopping=10
        )
        print(f"[{name}] Best params: {best_params}, AUC: {best_auc:.4f}")

        # Final runs and aggregation
        results = run_experiments_direct(
            ModelClass,
            best_params,
            train_x,
            train_y,
            train_z,
            test_x,
            test_y,
            num_runs=10,
            sample_size_per_class=250,
            early_stopping=10
        )
        aggregated = aggregate_metrics(results)

        # Save and display results
        write_results_csv(OUTPUT_CSV, name, aggregated, str(best_params))
        print_aggregated_results(name, aggregated, str(best_params))


if __name__ == "__main__":
    main()