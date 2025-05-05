# multitask_mlp.simul.py
import csv
import os
import random
from utils import (set_seed,grid_search_cv, aggregate_metrics, run_experiments,
                   plot_loss_curve, write_results_csv, print_aggregated_results)
from model.multitask_model.multitask_models import MultiTaskNN
from data.multitask_data import MultiTaskDatasetPreprocessor

set_seed(42)

CSV_FILENAME = "final_results.csv"
NUM_RUNS = 10

print("\n=== Multi-task Data Preprocessing (prompting/augmented_data.csv) ===")
preproc = MultiTaskDatasetPreprocessor(dataset_id=891, side_info_path='prompting/augmented_data.csv')
(grid_x, grid_main_y, grid_aux1, grid_aux2, grid_aux3,
 train_x, train_y, aux1_train, aux2_train, aux3_train,
 test_x, test_y) = preproc.preprocess()

param_grid = {
    "lr": [0.001],
    "hidden_dim": [256],
    "num_layers": [1],
    "lambda_aux": [0.01]
}

print("\n[Multi-task NN] Grid Search ...")
best_params, best_auc = grid_search_cv(
    MultiTaskNN, param_grid, grid_x, grid_main_y, cv=3, is_multitask=True,
    aux1=grid_aux1, aux2=grid_aux2, aux3=grid_aux3
)
print(f"Best Params: {best_params}, AUC: {best_auc:.4f}")

print("\n=== Multi-task Final Experiments ===")
results = run_experiments(
    MultiTaskNN, "MTL", NUM_RUNS, best_params,
    train_x, train_y, test_x, test_y,
    is_multitask=True, train_aux1_pool=aux1_train, train_aux2_pool=aux2_train, train_aux3_pool=aux3_train
)
agg = aggregate_metrics(results)

best_param_str = f"lr={best_params['lr']}, hidden_dim={best_params['hidden_dim']}, " \
                  f"num_layers={best_params['num_layers']}, lambda_aux={best_params['lambda_aux']}"

write_results_csv(CSV_FILENAME, "Multi-task MLP", agg, best_param_str)
print_aggregated_results("Multi-task MLP", agg, best_param_str)


def run_experiments(model_class, method_name, num_runs, best_params,
                    train_x_pool, train_y_pool, test_x_pool, test_y_pool,
                    optimizer_fixed="adam", epochs_fixed=300,
                    is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
    metrics_list = []
    class_counts = []
    train_df = pd.DataFrame(train_x_pool.numpy())
    train_df['Diabetes_binary'] = train_y_pool.numpy()
    class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
    class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
    test_pool_size = len(test_x_pool)
    
    init_kwargs = {}
    init_kwargs['input_dim'] = x_train.shape[1]
    # feed only the keys that actually appear in its __init__
    for k, v in best_params.items():
        if k in sig:
            init_kwargs[k] = v
    # model = model_class(**init_kwargs)
    # lr_val = best_params['lr']
    # hidden_dim_val = best_params['hidden_dim']
    # num_layers_val = best_params['num_layers']
    # lambda_aux_val = best_params['lambda_aux'] if 'lambda_aux' in best_params else 0.3
    print(class1_indices)
    for run_idx in range(num_runs):
        sampled1 = random.sample(class1_indices, 250)
        # print(sampled1)
        sampled0 = random.sample(class0_indices, 250)
        train_idx = sampled1 + sampled0
        x_train_run = train_x_pool[train_idx]
        y_train_run = train_y_pool[train_idx]
        n_pos = int((y_train_run == 1).sum().item())
        n_neg = int((y_train_run == 0).sum().item())
        class_counts.append({
            'run': run_idx+1,
            'n_neg': n_neg,
            'n_pos': n_pos
        })
        print(f"Run {run_idx+1}: negative={n_neg}, positive={n_pos}")

        if is_multitask:
            aux1_train_run = train_aux1_pool[train_idx]
            aux2_train_run = train_aux2_pool[train_idx]
            aux3_train_run = train_aux3_pool[train_idx]
        # ─────────────────────────────────────────────────────────        
        # Randomly select 125 samples from the test pool
        test_indices = random.sample(range(test_pool_size), 125)
        x_test_run = test_x_pool[test_indices]
        y_test_run = test_y_pool[test_indices]
        print(test_indices)
        model = model_class(**init_kwargs)
        # model = model_class(
        #     input_dim=x_train_run.shape[1],
        #     hidden_dim=hidden_dim_val,
        #     num_layers=num_layers_val,
        #     optimizer_type=optimizer_fixed,
        #     lr=lr_val,
        #     epochs=epochs_fixed,
        #     lambda_aux=lambda_aux_val
        #     )
        
        # Split training run data into training and validation sets (80/20 split)
        indices = list(range(len(x_train_run)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]
        x_train_sub = x_train_run[train_idx]
        y_train_sub = y_train_run[train_idx]
        x_val_sub = x_train_run[val_idx]
        y_val_sub = y_train_run[val_idx]
        
        if is_multitask:
            aux1_train_sub = aux1_train_run[train_idx]
            aux2_train_sub = aux2_train_run[train_idx]
            aux3_train_sub = aux3_train_run[train_idx]
            aux1_val_sub = aux1_train_run[val_idx]
            aux2_val_sub = aux2_train_run[val_idx]
            aux3_val_sub = aux3_train_run[val_idx]
            loss_data = model.train_model(x_train_sub, y_train_sub, aux1_train_sub, aux2_train_sub, aux3_train_sub,
                                          x_val=x_val_sub, main_y_val=y_val_sub, aux1_val=aux1_val_sub, aux2_val=aux2_val_sub, aux3_val=aux3_val_sub,
                                          early_stopping_patience=30, record_loss=True)
        else:
            loss_data = model.train_model(x_train_sub, y_train_sub,
                                          x_val=x_val_sub, y_val=y_val_sub,
                                          early_stopping_patience=30, record_loss=True)
        train_losses, val_losses = loss_data
        
        if not os.path.exists("img"):
            os.makedirs("img")
        plot_filename = os.path.join("img", f"MT+MLP_7B_wosplit{run_idx+1}.png")
        plt.figure(figsize=(8,6))
        plt.plot(train_losses, label="Train BCE Loss")
        plt.plot(val_losses, label="Validation BCE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Train vs Validation BCE Loss")
        plt.legend()
        plt.savefig(plot_filename)
        plt.show()
        
        # Evaluate model on test set
        metrics = model.evaluate(x_test_run, y_test_run)
        metrics_list.append(metrics)
        print(metrics)
    return metrics_list

    