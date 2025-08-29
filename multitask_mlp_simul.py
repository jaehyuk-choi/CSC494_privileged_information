# # multitask_mlp.simul.py
# import csv
# import os
# import random
# from utils import (set_seed,grid_search_cv, aggregate_metrics, run_experiments,
#                    plot_loss_curve, write_results_csv, print_aggregated_results)
# from model.multitask_model.multitask_models import MultiTaskNN
# from data.multitask_data import MultiTaskDatasetPreprocessor

# set_seed(42)

# CSV_FILENAME = "final_results.csv"
# NUM_RUNS = 10

# print("\n=== Multi-task Data Preprocessing (prompting/augmented_data.csv) ===")
# preproc = MultiTaskDatasetPreprocessor(dataset_id=891, side_info_path='prompting/augmented_data.csv')
# (grid_x, grid_main_y, grid_aux1, grid_aux2, grid_aux3,
#  train_x, train_y, aux1_train, aux2_train, aux3_train,
#  test_x, test_y) = preproc.preprocess()

# param_grid = {
#     "lr": [0.001],
#     "hidden_dim": [256],
#     "num_layers": [1],
#     "lambda_aux": [0.01]
# }

# print("\n[Multi-task NN] Grid Search ...")
# best_params, best_auc = grid_search_cv(
#     MultiTaskNN, param_grid, grid_x, grid_main_y, cv=3, is_multitask=True,
#     aux1=grid_aux1, aux2=grid_aux2, aux3=grid_aux3
# )
# print(f"Best Params: {best_params}, AUC: {best_auc:.4f}")

# print("\n=== Multi-task Final Experiments ===")
# results = run_experiments(
#     MultiTaskNN, "MTL", NUM_RUNS, best_params,
#     train_x, train_y, test_x, test_y,
#     is_multitask=True, train_aux1_pool=aux1_train, train_aux2_pool=aux2_train, train_aux3_pool=aux3_train
# )
# agg = aggregate_metrics(results)

# best_param_str = f"lr={best_params['lr']}, hidden_dim={best_params['hidden_dim']}, " \
#                   f"num_layers={best_params['num_layers']}, lambda_aux={best_params['lambda_aux']}"

# write_results_csv(CSV_FILENAME, "Multi-task MLP", agg, best_param_str)
# print_aggregated_results("Multi-task MLP", agg, best_param_str)


# def run_experiments(model_class, method_name, num_runs, best_params,
#                     train_x_pool, train_y_pool, test_x_pool, test_y_pool,
#                     optimizer_fixed="adam", epochs_fixed=300,
#                     is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
#     metrics_list = []
#     class_counts = []
#     train_df = pd.DataFrame(train_x_pool.numpy())
#     train_df['Diabetes_binary'] = train_y_pool.numpy()
#     class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
#     class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
#     test_pool_size = len(test_x_pool)
    
#     init_kwargs = {}
#     init_kwargs['input_dim'] = x_train.shape[1]
#     # feed only the keys that actually appear in its __init__
#     for k, v in best_params.items():
#         if k in sig:
#             init_kwargs[k] = v
#     # model = model_class(**init_kwargs)
#     # lr_val = best_params['lr']
#     # hidden_dim_val = best_params['hidden_dim']
#     # num_layers_val = best_params['num_layers']
#     # lambda_aux_val = best_params['lambda_aux'] if 'lambda_aux' in best_params else 0.3
#     print(class1_indices)
#     for run_idx in range(num_runs):
#         sampled1 = random.sample(class1_indices, 250)
#         # print(sampled1)
#         sampled0 = random.sample(class0_indices, 250)
#         train_idx = sampled1 + sampled0
#         x_train_run = train_x_pool[train_idx]
#         y_train_run = train_y_pool[train_idx]
#         n_pos = int((y_train_run == 1).sum().item())
#         n_neg = int((y_train_run == 0).sum().item())
#         class_counts.append({
#             'run': run_idx+1,
#             'n_neg': n_neg,
#             'n_pos': n_pos
#         })
#         print(f"Run {run_idx+1}: negative={n_neg}, positive={n_pos}")

#         if is_multitask:
#             aux1_train_run = train_aux1_pool[train_idx]
#             aux2_train_run = train_aux2_pool[train_idx]
#             aux3_train_run = train_aux3_pool[train_idx]
#         # ─────────────────────────────────────────────────────────        
#         # Randomly select 125 samples from the test pool
#         test_indices = random.sample(range(test_pool_size), 125)
#         x_test_run = test_x_pool[test_indices]
#         y_test_run = test_y_pool[test_indices]
#         print(test_indices)
#         model = model_class(**init_kwargs)
#         # model = model_class(
#         #     input_dim=x_train_run.shape[1],
#         #     hidden_dim=hidden_dim_val,
#         #     num_layers=num_layers_val,
#         #     optimizer_type=optimizer_fixed,
#         #     lr=lr_val,
#         #     epochs=epochs_fixed,
#         #     lambda_aux=lambda_aux_val
#         #     )
        
#         # Split training run data into training and validation sets (80/20 split)
#         indices = list(range(len(x_train_run)))
#         random.shuffle(indices)
#         split = int(0.8 * len(indices))
#         train_idx, val_idx = indices[:split], indices[split:]
#         x_train_sub = x_train_run[train_idx]
#         y_train_sub = y_train_run[train_idx]
#         x_val_sub = x_train_run[val_idx]
#         y_val_sub = y_train_run[val_idx]
        
#         if is_multitask:
#             aux1_train_sub = aux1_train_run[train_idx]
#             aux2_train_sub = aux2_train_run[train_idx]
#             aux3_train_sub = aux3_train_run[train_idx]
#             aux1_val_sub = aux1_train_run[val_idx]
#             aux2_val_sub = aux2_train_run[val_idx]
#             aux3_val_sub = aux3_train_run[val_idx]
#             loss_data = model.train_model(x_train_sub, y_train_sub, aux1_train_sub, aux2_train_sub, aux3_train_sub,
#                                           x_val=x_val_sub, main_y_val=y_val_sub, aux1_val=aux1_val_sub, aux2_val=aux2_val_sub, aux3_val=aux3_val_sub,
#                                           early_stopping_patience=30, record_loss=True)
#         else:
#             loss_data = model.train_model(x_train_sub, y_train_sub,
#                                           x_val=x_val_sub, y_val=y_val_sub,
#                                           early_stopping_patience=30, record_loss=True)
#         train_losses, val_losses = loss_data
        
#         if not os.path.exists("img"):
#             os.makedirs("img")
#         plot_filename = os.path.join("img", f"MT+MLP_7B_wosplit{run_idx+1}.png")
#         plt.figure(figsize=(8,6))
#         plt.plot(train_losses, label="Train BCE Loss")
#         plt.plot(val_losses, label="Validation BCE Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("BCE Loss")
#         plt.title("Train vs Validation BCE Loss")
#         plt.legend()
#         plt.savefig(plot_filename)
#         plt.show()
        
#         # Evaluate model on test set
#         metrics = model.evaluate(x_test_run, y_test_run)
#         metrics_list.append(metrics)
#         print(metrics)
#     return metrics_list

    
import os
import csv
import itertools
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ucimlrepo import fetch_ucirepo

# =============================================================================
# 1. Set Random Seed for Reproducibility
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# 2. Data Preprocessing Class for the Multi-Task Setting
#    (Uses an augmented CSV file along with UCI data)
# =============================================================================
class MultiTaskDatasetPreprocessor:
    """
    Preprocessor for the multi-task model.
    
    This class loads data from the UCI repository (using fetch_ucirepo)
    and an augmented CSV file. The CSV file includes a set of training
    features and four target columns:
      - 'Diabetes_binary' (the main target)
      - 'health_1_10' (a continuous auxiliary target used as side info)
      - 'diabetes_risk_score' (auxiliary regression target)
      - 'has_diabetes' (auxiliary binary target)
      
    It then creates a balanced grid for hyperparameter search, a training
    pool, and a test pool. Finally, the data are converted to PyTorch tensors.
    """
    def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data_70B.csv'):
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features  # pandas DataFrame
        self.y_original = self.original_data.data.targets   # pandas Series
        self.side_info_path = side_info_path
        self.scaler = StandardScaler()
        
    def preprocess(self):
        # In this example, we use a subset of training columns.
        training_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                           'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                           'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                           'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        # Target columns: the four targets
        target_cols = ['has_diabetes', 'health_1_10', 'diabetes_risk_score', 'Diabetes_binary']
        augmented_df = pd.read_csv(self.side_info_path)
        augmented_df = augmented_df[training_cols + target_cols]
        
        # Fill missing values for targets
        augmented_df['has_diabetes'].fillna(0, inplace=True)
        augmented_df['health_1_10'].fillna(augmented_df['health_1_10'].median(), inplace=True)
        augmented_df['diabetes_risk_score'].fillna(augmented_df['diabetes_risk_score'].median(), inplace=True)
        
        # Create a balanced grid for hyperparameter search
        pos_idx = augmented_df[augmented_df['Diabetes_binary'] == 1].index.tolist()
        neg_idx = augmented_df[augmented_df['Diabetes_binary'] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
        grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
        grid_df = pd.concat([grid_pos, grid_neg])
        
        # Training Pool: remaining data
        train_pool_df = augmented_df.drop(index=grid_df.index)
        
        # Test Pool: remove indices in augmented_df from original UCI data
        original_df = self.X_original.copy()
        original_df['Diabetes_binary'] = self.y_original
        test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')
        
        # (Optional) Scaling can be added here if needed.
        grid_df_scaled = grid_df.copy()
        train_pool_df_scaled = train_pool_df.copy()
        test_pool_df_scaled = test_pool_df.copy()
        
        def df_to_tensors_multi(df):
            # Convert DataFrame to PyTorch tensors.
            features = df.drop(columns=target_cols, errors='ignore')
            x = torch.from_numpy(features.values).float()
            y_main = torch.from_numpy(df['Diabetes_binary'].values).float()  # main target
            y_aux1 = torch.from_numpy(df['health_1_10'].values).float()       # auxiliary (will also be used as side info)
            y_aux2 = torch.from_numpy(df['diabetes_risk_score'].values).float() # auxiliary
            y_aux3 = torch.from_numpy(df['has_diabetes'].values).float()        # auxiliary
            return x, y_main, y_aux1, y_aux2, y_aux3
        
        def df_to_tensors_test(df):
            # For test pool, use only training columns and main target.
            features = df[training_cols]
            x = torch.from_numpy(features.values).float()
            y = torch.from_numpy(df['Diabetes_binary'].values).float()
            return x, y
        
        grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3 = df_to_tensors_multi(grid_df_scaled)
        train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool = df_to_tensors_multi(train_pool_df_scaled)
        test_x_pool, test_y_pool = df_to_tensors_test(test_pool_df_scaled)
        
        return (grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3,
                train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
                test_x_pool, test_y_pool)

# =============================================================================
# 3. Combined Direct + Multi-Task Model
# 
# This model computes an intermediate representation S from input X
# using several encoder layers. It then:
#  (a) Computes a direct loss: forces S to be similar to a chosen side-
#      information target (here we use the auxiliary target 'health_1_10').
#  (b) Predicts the main target Y via a head ψ(S).
#  (c) Predicts three auxiliary targets in parallel via separate heads.
#
# The total loss is:
#      L_total = L_main + λ_direct * L_direct + λ_aux * (L_aux1 + L_aux2 + L_aux3)
# =============================================================================
# Inside CombinedMultiTaskNN class
class CombinedMultiTaskNN(nn.Module):
    def __init__(self, input_size, hidden_dim=16, num_layers=1,
                 lr=0.01, lambda_aux=0.3, lambda_direct=0.3, epochs=300):
        super(CombinedMultiTaskNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lambda_aux = lambda_aux
        self.lambda_direct = lambda_direct
        
        # Feature extractor: Encoder layers
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(input_size, hidden_dim),
                    nn.ReLU()
                ))
            else:
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ))
        
        # Shared layer for main and auxiliary tasks
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Main task head (binary classification)
        self.main_task = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary task heads
        self.aux_task1 = nn.Linear(hidden_dim, 1)
        self.aux_task2 = nn.Linear(hidden_dim, 1)
        self.aux_task3 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Compress S to scalar for direct loss (forcing S ~ side info)
        self.to_scalar = nn.Linear(hidden_dim, 1)
        
        # **Learnable weights for weighted-sum of intermediate outputs**
        self.alpha_aux1 = nn.Parameter(torch.ones(num_layers))
        self.alpha_aux2 = nn.Parameter(torch.ones(num_layers))
        self.alpha_aux3 = nn.Parameter(torch.ones(num_layers))
        
        # Loss functions
        self.main_loss_fn = nn.BCELoss()
        self.aux_loss_fn = nn.MSELoss()
        self.aux_task3_loss_fn = nn.BCELoss()
        self.direct_loss_fn = nn.MSELoss()
        
        # Optimizer (will be used during training)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)


    def forward(self, x):
        # Encoder pass with intermediate outputs
        intermediate_outputs = []
        h = x
        for layer in self.feature_layers:
            h = layer(h)
            intermediate_outputs.append(h)
        
        # Final intermediate representation S
        s = intermediate_outputs[-1]
        s_scalar = self.to_scalar(s).view(-1)  # Compress S to a scalar

        # Main task head
        h_main = self.shared(s)
        main_out = self.main_task(h_main).view(-1)
        
        # Auxiliary tasks: Weighted sum of intermediate outputs
        def weighted_sum(alpha, outputs):
            weights = torch.softmax(alpha, dim=0)
            combined = sum(w * out for w, out in zip(weights, outputs))
            return combined
        
        h_aux1 = self.shared(weighted_sum(self.alpha_aux1, intermediate_outputs))
        h_aux2 = self.shared(weighted_sum(self.alpha_aux2, intermediate_outputs))
        h_aux3 = self.shared(weighted_sum(self.alpha_aux3, intermediate_outputs))
        
        aux1_out = self.aux_task1(h_aux1).view(-1)
        aux2_out = self.aux_task2(h_aux2).view(-1)
        aux3_out = self.aux_task3(h_aux3).view(-1)
        
        return s_scalar, main_out, aux1_out, aux2_out, aux3_out

    def compute_loss(self, x, main_y, aux1_y, aux2_y, aux3_y):
        """
        Compute losses:
          - Main task BCE loss between main_out and main_y.
          - Direct loss: MSE between compressed S and side-information target.
          - Auxiliary losses for aux_task1, aux_task2, and aux_task3.
        """
        s_scalar, main_out, aux1_out, aux2_out, aux3_out = self.forward(x)
        
        # Main task loss
        loss_main = self.main_loss_fn(main_out, main_y.view(-1))
        
        # Direct loss between S_scalar and side information (aux1_y)
        loss_direct = self.direct_loss_fn(s_scalar, aux1_y.view(-1))  # <-- FIXED HERE
        
        # Auxiliary losses
        loss_aux1 = self.aux_loss_fn(aux1_out, aux1_y.view(-1))
        loss_aux2 = self.aux_loss_fn(aux2_out, aux2_y.view(-1))
        loss_aux3 = self.aux_task3_loss_fn(aux3_out, aux3_y.view(-1))
        
        # Total loss
        total_loss = loss_main + self.lambda_direct * loss_direct + self.lambda_aux * (loss_aux1 + loss_aux2 + loss_aux3)
        return total_loss, loss_main, loss_direct, loss_aux1, loss_aux2, loss_aux3


    def train_model(self, x, main_y, aux1_y, aux2_y, aux3_y,
                    x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
                    early_stopping_patience=10, record_loss=False):
        """
        Train the model on training data (x and targets). If validation data are provided,
        compute the main task loss on the validation set each epoch and use early stopping.
        If record_loss is True, return lists of main task losses for train and val.
        """
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            total_loss, loss_main, loss_direct, loss_aux1, loss_aux2, loss_aux3 = self.compute_loss(
                x, main_y, aux1_y, aux2_y, aux3_y)
            total_loss.backward()
            self.optimizer.step()
            if record_loss:
                train_loss_history.append(loss_main.item())
            if x_val is not None and main_y_val is not None:
                self.eval()
                with torch.no_grad():
                    _, val_loss_main, _, _, _, _ = self.compute_loss(
                        x_val, main_y_val, aux1_val, aux2_val, aux3_val)
                if record_loss:
                    val_loss_history.append(val_loss_main.item())
                if val_loss_main.item() < best_val_loss:
                    best_val_loss = val_loss_main.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break
        if record_loss:
            return train_loss_history, val_loss_history

    def evaluate(self, x, y):
        """
        Evaluate the model on a test set.
        Returns a dictionary of metrics: AUC, class-wise precision/recall/F1, and test BCE loss.
        """
        self.eval()
        with torch.no_grad():
            _, y_pred, _, _, _ = self.forward(x)
            preds = (y_pred >= 0.5).float()
            metrics = precision_recall_fscore_support(
                y.cpu().numpy(), preds.cpu().numpy(), labels=[0,1], zero_division=0
            )
            try:
                auc = roc_auc_score(y.cpu().numpy(), y_pred.cpu().numpy())
            except Exception:
                auc = 0.0
            bce_loss = self.main_loss_fn(y_pred, y.view(-1))
        return {
            "auc": auc,
            "p0": metrics[0][0],
            "r0": metrics[1][0],
            "f0": metrics[2][0],
            "p1": metrics[0][1],
            "r1": metrics[1][1],
            "f1": metrics[2][1],
            "bce_loss": bce_loss.item()
        }

# =============================================================================
# 4. Grid Search Function (based on ROC AUC for the main task)
# =============================================================================
def grid_search_cv(model_class, param_grid, X, y, cv=3, is_multitask=False,
                   aux1=None, aux2=None, aux3=None, lr_default=0.01):
    best_auc = -1.0
    best_params = None
    param_keys = list(param_grid.keys())
    kf = KFold(n_splits=cv, shuffle=True, random_state=SEED)
    for combo in itertools.product(*(param_grid[k] for k in param_keys)):
        combi = dict(zip(param_keys, combo))
        aucs = []
        for train_idx, val_idx in kf.split(X):
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            if is_multitask:
                aux1_train = aux1[train_idx]
                aux2_train = aux2[train_idx]
                aux3_train = aux3[train_idx]
                aux1_val = aux1[val_idx]
                aux2_val = aux2[val_idx]
                aux3_val = aux3[val_idx]
            model = model_class(
                input_size=X_train.shape[1],
                hidden_dim=combi.get('hidden_dim', 32),
                num_layers=combi.get('num_layers', 1),
                lr=combi.get('lr', lr_default),
                epochs=100,  # shorter training for grid search
                lambda_aux=combi.get('lambda_aux', 0.3),
                lambda_direct=combi.get('lambda_direct', 0.3)
            )
            if is_multitask:
                model.train_model(X_train, y_train, aux1_train, aux2_train, aux3_train,
                                  x_val=X_val, main_y_val=y_val, aux1_val=aux1_val, aux2_val=aux2_val, aux3_val=aux3_val,
                                  early_stopping_patience=10, record_loss=False)
            else:
                model.train_model(X_train, y_train, x_val=X_val, main_y_val=y_val,
                                  early_stopping_patience=10, record_loss=False)
            model.eval()
            with torch.no_grad():
                _, y_val_pred, _, _, _ = model.forward(X_val)
            try:
                auc_val = roc_auc_score(y_val.cpu().numpy(), y_val_pred.cpu().numpy())
            except Exception:
                auc_val = 0.0
            aucs.append(auc_val)
        avg_auc = np.mean(aucs)
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_params = combi
    return best_params, best_auc

# =============================================================================
# 5. Run Experiments Function (run model multiple times with loss plotting)
# =============================================================================
def run_experiments(model_class, method_name, num_runs, best_params,
                    train_x_pool, train_y_pool, test_x_pool, test_y_pool,
                    optimizer_fixed="adam", epochs_fixed=300,
                    is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
    metrics_list = []
    # Create a DataFrame from train_x_pool for balanced sampling
    train_df = pd.DataFrame(train_x_pool.numpy())
    train_df['Diabetes_binary'] = train_y_pool.numpy()
    class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
    class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
    test_pool_size = len(test_x_pool)
    
    lr_val = best_params['lr']
    hidden_dim_val = best_params['hidden_dim']
    num_layers_val = best_params['num_layers']
    lambda_aux_val = best_params['lambda_aux'] if 'lambda_aux' in best_params else 0.3
    lambda_direct_val = best_params['lambda_direct'] if 'lambda_direct' in best_params else 0.3
    
    for run_idx in range(num_runs):
        print(f"[{method_name}] Run {run_idx+1}/{num_runs}")
        # Randomly sample 250 instances from each class (total 500 samples)
        sampled_class1 = random.sample(class1_indices, 250)
        sampled_class0 = random.sample(class0_indices, 250)
        balanced_indices = sampled_class1 + sampled_class0
        
        x_train_run = train_x_pool[balanced_indices]
        y_train_run = train_y_pool[balanced_indices]
        if is_multitask:
            aux1_train_run = train_aux1_pool[balanced_indices]
            aux2_train_run = train_aux2_pool[balanced_indices]
            aux3_train_run = train_aux3_pool[balanced_indices]
        
        # Randomly select 125 samples from the test pool
        test_indices = random.sample(range(test_pool_size), 125)
        x_test_run = test_x_pool[test_indices]
        y_test_run = test_y_pool[test_indices]
        
        model = model_class(
            input_size=x_train_run.shape[1],
            hidden_dim=hidden_dim_val,
            num_layers=num_layers_val,
            lr=lr_val,
            epochs=epochs_fixed,
            lambda_aux=lambda_aux_val,
            lambda_direct=lambda_direct_val
        )
        
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
                                          early_stopping_patience=10, record_loss=True)
        else:
            loss_data = model.train_model(x_train_sub, y_train_sub,
                                          x_val=x_val_sub, main_y_val=y_val_sub,
                                          early_stopping_patience=10, record_loss=True)
        train_losses, val_losses = loss_data
        
        # Plot training vs. validation loss for this run
        if not os.path.exists("img"):
            os.makedirs("img")
        plot_filename = os.path.join("img", f"{method_name}_Run{run_idx+1}.png")
        plt.figure(figsize=(8,6))
        plt.plot(train_losses, label="Train Main Loss")
        plt.plot(val_losses, label="Validation Main Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{method_name} - Run {run_idx+1}")
        plt.legend()
        plt.savefig(plot_filename)
        plt.close()
        
        # Evaluate the model on the test set
        metrics = model.evaluate(x_test_run, y_test_run)
        metrics_list.append(metrics)
    return metrics_list

# =============================================================================
# 6. Aggregation Function: Aggregate results as "mean ± std"
# =============================================================================
def aggregate_metrics(metrics_list):
    agg = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        agg[key] = f"{np.mean(values):.2f} ± {np.std(values):.2f}"
    return agg

# =============================================================================
# 7. Main Function: Data Preprocessing, Grid Search, and 10 Iterations Run
# =============================================================================
def main():
    CSV_FILENAME = "final_results.csv"
    NUM_RUNS = 10

    print("\n=== Multi-task Data Preprocessing (prompting/augmented_data_70B.csv) ===")
    preproc = MultiTaskDatasetPreprocessor(dataset_id=891, side_info_path='prompting/augmented_data.csv')
    (grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3,
     mt_train_x, mt_train_main_y, mt_train_aux1, mt_train_aux2, mt_train_aux3,
     mt_test_x, mt_test_y) = preproc.preprocess()
    
    # For our combined model, we use:
    #   main_y: Diabetes_binary (grid_y)
    #   aux1: health_1_10 (also used as side info for direct loss)
    #   aux2: diabetes_risk_score
    #   aux3: has_diabetes
    # Define grid search hyperparameter grid.
    param_grid = {
        "lr": [0.01],
        "hidden_dim": [64,128,256],
        "num_layers": [1,2,3,4],
        "lambda_aux": [0.01, 0.1, 0.3],
        "lambda_direct": [0.01, 0.1, 0.3]
    }
    print("\n[Combined Multi-task NN] Grid Search ...")
    best_params, best_auc = grid_search_cv(CombinedMultiTaskNN, param_grid,
                                             grid_x, grid_y, cv=3, is_multitask=True,
                                             aux1=grid_aux1, aux2=grid_aux2, aux3=grid_aux3)
    print(f"Combined NN best params: {best_params}, AUC: {best_auc:.4f}")
    
    # Run final experiments (10 iterations) for the combined model
    print("\n=== Combined Model Final Experiments ===")
    results = run_experiments(CombinedMultiTaskNN, "CombinedMultiTaskNN", NUM_RUNS, best_params,
                              mt_train_x, mt_train_main_y, mt_test_x, mt_test_y,
                              optimizer_fixed="adam", epochs_fixed=300,
                              is_multitask=True, train_aux1_pool=mt_train_aux1, train_aux2_pool=mt_train_aux2, train_aux3_pool=mt_train_aux3)
    agg_metrics = aggregate_metrics(results)
    
    best_hyperparams_str = (f"lr: {best_params['lr']}, hidden_dim: {best_params['hidden_dim']}, "
                            f"num_layers: {best_params['num_layers']}, lambda_aux: {best_params['lambda_aux']}, "
                            f"lambda_direct: {best_params['lambda_direct']}")
    
    # Save final aggregated results to CSV
    # with open(CSV_FILENAME, mode='a', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Method", "AUC", "Precision_class0", "Recall_class0", "F1_class0", 
    #                      "Precision_class1", "Recall_class1", "F1_class1", "Test_BCE_Loss", 
    #                      "Best Parameter Set"])
    #     method = "Combined Multi-task NN"
    #     row = [
    #         method,
    #         agg_metrics.get("auc", ""),
    #         agg_metrics.get("p0", ""),
    #         agg_metrics.get("r0", ""),
    #         agg_metrics.get("f0", ""),
    #         agg_metrics.get("p1", ""),
    #         agg_metrics.get("r1", ""),
    #         agg_metrics.get("f1", ""),
    #         agg_metrics.get("bce_loss", ""),
    #         best_hyperparams_str
    #     ]
        # writer.writerow(row)
    
    print("\n=== Final Aggregated Results (one line per model) ===")
    print("Method: Combined Multi-task with Direct model")
    print(f"  AUC: {agg_metrics.get('auc', '')}")
    print(f"  Class0 -> Precision: {agg_metrics.get('p0', '')}, Recall: {agg_metrics.get('r0', '')}, F1: {agg_metrics.get('f0', '')}")
    print(f"  Class1 -> Precision: {agg_metrics.get('p1', '')}, Recall: {agg_metrics.get('r1', '')}, F1: {agg_metrics.get('f1', '')}")
    print(f"  Test BCE Loss: {agg_metrics.get('bce_loss', 'N/A')}")
    print(f"  Best Parameter Set: {best_hyperparams_str}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    main() 