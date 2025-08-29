# # multitask_project.py

# # =========================
# #        IMPORTS
# # =========================

# import os
# import csv
# import random
# import itertools
# import inspect

# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, log_loss

# from ucimlrepo import fetch_ucirepo


# # =========================
# #   GLOBAL SEED FUNCTION
# # =========================

# def set_seed(seed: int):
#     """
#     Set the random seed for reproducibility across random, NumPy, and PyTorch.
#     """
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# # =========================
# #   DATA PREPROCESSOR
# # =========================

# class MultiTaskDatasetPreprocessor:
#     """
#     Preprocessor for the multi-task model.
#     Combines LLM-augmented data with original UCI data.
#     Produces train/test splits and auxiliary targets for multi-task learning.
#     """

#     def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
#         """
#         Args:
#             dataset_id (int): ID of the UCI dataset to fetch (default: 891).
#             side_info_path (str): Path to CSV containing LLM-augmented side information.
#         """
#         # Fetch original UCI dataset
#         self.original_data = fetch_ucirepo(id=dataset_id)
#         self.X_original = self.original_data.data.features  # pandas DataFrame
#         self.y_original = self.original_data.data.targets   # pandas Series

#         # Standardizer for continuous features
#         self.scaler = StandardScaler()

#         # Path to side information CSV
#         self.side_info_path = side_info_path

#         # Columns used as input features
#         self.training_cols = [
#             'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
#             'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#             'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
#             'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
#             'Income'
#         ]
#         # Columns for main and auxiliary targets
#         self.target_cols = ['has_diabetes', 'health_1_10', 'diabetes_risk_score', 'Diabetes_binary']

#     def preprocess(self):
#         """
#         Execute preprocessing and return tensors for:
#           - grid set (balanced) for hyperparameter tuning
#           - training pool with auxiliary targets
#           - test set (only main task)
        
#         Returns:
#             tuple: (grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3,
#                     train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
#                     test_x_pool, test_y_pool)
#         """
#         continuous_cols = ['BMI']

#         # 1) Load the augmented dataset with side information
#         augmented_df = pd.read_csv(self.side_info_path)
#         augmented_df = augmented_df[self.training_cols + self.target_cols]

#         # 2) Fill or drop missing values in auxiliary and main targets
#         augmented_df['has_diabetes'].fillna(0, inplace=True)
#         augmented_df['health_1_10'].fillna(augmented_df['health_1_10'].median(), inplace=True)
#         augmented_df['diabetes_risk_score'].fillna(augmented_df['diabetes_risk_score'].median(), inplace=True)
#         augmented_df['Diabetes_binary'].fillna(0, inplace=True)

#         # 3) Create a balanced 'grid' set from the augmented data for hyperparameter tuning
#         pos_idx = augmented_df[augmented_df['Diabetes_binary'] == 1].index.tolist()
#         neg_idx = augmented_df[augmented_df['Diabetes_binary'] == 0].index.tolist()
#         n_pos = min(len(pos_idx), 375)
#         n_neg = min(len(neg_idx), 375)
#         grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
#         grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
#         grid_df = pd.concat([grid_pos, grid_neg])

#         # Remaining training data used as 'train pool'
#         train_pool = augmented_df.drop(index=grid_df.index)

#         # 4) Construct a separate test set using original dataset (no augmented info)
#         original_df = self.X_original.copy()
#         original_df['Diabetes_binary'] = self.y_original
#         # Exclude rows that appear in augmented_df to avoid overlap
#         test_pool = original_df.drop(index=augmented_df.index, errors='ignore')
#         test_pool = test_pool[self.training_cols + ['Diabetes_binary']]

#         # 5) Fit scaler on continuous columns of train_pool, then transform grid, train_pool, test_pool
#         self.scaler.fit(train_pool[continuous_cols])
#         grid_df_scaled = grid_df.copy()
#         grid_df_scaled[continuous_cols] = self.scaler.transform(grid_df[continuous_cols])
#         train_pool_scaled = train_pool.copy()
#         train_pool_scaled[continuous_cols] = self.scaler.transform(train_pool[continuous_cols])
#         test_pool_scaled = test_pool.copy()
#         test_pool_scaled[continuous_cols] = self.scaler.transform(test_pool[continuous_cols])

#         # 6) Convert DataFrames to PyTorch tensors (multi-task)
#         def df_to_tensors_multi(df):
#             features = df.drop(columns=self.target_cols, errors='ignore')
#             x = torch.from_numpy(features.values).float()
#             y_main = torch.from_numpy(df['Diabetes_binary'].values).float()
#             y_aux1 = torch.from_numpy(df['health_1_10'].values).float()
#             y_aux2 = torch.from_numpy(df['diabetes_risk_score'].values).float()
#             y_aux3 = torch.from_numpy(df['has_diabetes'].values).float()
#             return x, y_main, y_aux1, y_aux2, y_aux3

#         # Convert grid set
#         grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3 = df_to_tensors_multi(grid_df_scaled)

#         # Convert train pool
#         train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool = df_to_tensors_multi(train_pool_scaled)

#         # 7) Convert test set to tensors (only main task)
#         def df_to_tensors_test(df):
#             features = df[self.training_cols]
#             x = torch.from_numpy(features.values).float()
#             y = torch.from_numpy(df['Diabetes_binary'].values).float()
#             return x, y

#         test_x_pool, test_y_pool = df_to_tensors_test(test_pool_scaled)

#         return (
#             grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3,
#             train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
#             test_x_pool, test_y_pool
#         )


# # =========================
# #       MODEL CLASSES
# # =========================

# SEED = 42  # Global seed for PyTorch reproducibility


# class MultiTaskLogisticRegression(nn.Module):
#     """
#     Multi-task Logistic Regression:
#       - Shares a linear layer for feature transformation + batch norm.
#       - Main head: binary classification with BCE loss.
#       - Aux1 & Aux2: regression tasks with MSE loss.
#       - Aux3: binary auxiliary task with BCE loss.
#       - Total loss = main_bce + λ1*MSE(aux1) + λ2*MSE(aux2) + λ3*BCE(aux3).
#     """

#     def __init__(self, input_dim, optimizer_type="adam", lr=0.01,
#                  lambda_aux1=0.3, lambda_aux2=0.3, lambda_aux3=0.3,
#                  epochs=300):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             optimizer_type (str): "adam" or "sgd" for optimizer selection.
#             lr (float): Learning rate.
#             lambda_aux1 (float): Weight for aux1 (MSE) loss.
#             lambda_aux2 (float): Weight for aux2 (MSE) loss.
#             lambda_aux3 (float): Weight for aux3 (BCE) loss.
#             epochs (int): Number of training epochs.
#         """
#         super().__init__()
#         # Shared linear + batch norm layer
#         self.shared = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.BatchNorm1d(input_dim)
#         )

#         # Main task head (binary classification)
#         self.main_head = nn.Sequential(
#             nn.Linear(input_dim, 1),
#             nn.Sigmoid()
#         )

#         # Auxiliary heads
#         self.aux1 = nn.Linear(input_dim, 1)                 # regression
#         self.aux2 = nn.Linear(input_dim, 1)                 # regression
#         self.aux3 = nn.Sequential(nn.Linear(input_dim, 1),  # binary
#                                    nn.Sigmoid())

#         self.lr = lr
#         self.epochs = epochs
#         self.lambda_aux1 = lambda_aux1
#         self.lambda_aux2 = lambda_aux2
#         self.lambda_aux3 = lambda_aux3

#         # Choose optimizer
#         if optimizer_type.lower() == "adam":
#             self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
#         else:
#             self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

#         # Loss functions
#         self.main_loss_fn = nn.BCELoss()
#         self.aux_mse = nn.MSELoss()
#         self.aux_bce = nn.BCELoss()

#     def forward(self, x):
#         """
#         Forward pass returning main_out, aux1_out, aux2_out, aux3_out.
#         """
#         feat = self.shared(x)
#         main = self.main_head(feat).view(-1)
#         a1 = self.aux1(feat).view(-1)
#         a2 = self.aux2(feat).view(-1)
#         a3 = self.aux3(feat).view(-1)
#         return main, a1, a2, a3

#     def train_model(self, x, main_y, aux1_y=None, aux2_y=None, aux3_y=None,
#                     x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
#                     early_stopping_patience=30, record_loss=False):
#         """
#         Train the multi-task LR model.
#         Args:
#             x (Tensor): Training features.
#             main_y (Tensor): Main binary labels.
#             aux1_y (Tensor): Aux1 regression targets.
#             aux2_y (Tensor): Aux2 regression targets.
#             aux3_y (Tensor): Aux3 binary targets.
#             x_val (Tensor, optional): Validation features.
#             main_y_val (Tensor, optional): Validation main labels.
#             aux1_val, aux2_val, aux3_val: Validation aux targets.
#             early_stopping_patience (int): Patience for early stopping on main validation loss.
#             record_loss (bool): If True, returns (train_losses, val_losses) for the main loss.
#         Returns:
#             If record_loss: (train_losses, val_losses) lists of main-task BCE losses.
#         """
#         train_losses = []
#         val_losses = []
#         best_val = float('inf')
#         patience = 0

#         for epoch in range(self.epochs):
#             self.train()
#             self.optimizer.zero_grad()
#             main_out, a1_out, a2_out, a3_out = self.forward(x)

#             # Compute main loss (BCE)
#             loss_main = self.main_loss_fn(main_out, main_y.view(-1))

#             # Compute auxiliary losses if provided
#             if aux1_y is not None:
#                 loss_a1 = self.aux_mse(a1_out, aux1_y.view(-1))
#                 loss_a2 = self.aux_mse(a2_out, aux2_y.view(-1))
#                 loss_a3 = self.aux_bce(a3_out, aux3_y.view(-1))
#                 total_loss = (loss_main +
#                               self.lambda_aux1 * loss_a1 +
#                               self.lambda_aux2 * loss_a2 +
#                               self.lambda_aux3 * loss_a3)
#             else:
#                 total_loss = loss_main

#             total_loss.backward()
#             self.optimizer.step()

#             if record_loss:
#                 train_losses.append(loss_main.item())

#             # Validation step (for main task only)
#             if x_val is not None and main_y_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     val_main, _, _, _ = self.forward(x_val)
#                     v_loss = self.main_loss_fn(val_main, main_y_val.view(-1))
#                 if record_loss:
#                     val_losses.append(v_loss.item())

#                 # Early stopping on main validation loss
#                 if v_loss.item() < best_val:
#                     best_val = v_loss.item()
#                     patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping_patience:
#                         break

#         if record_loss:
#             return train_losses, val_losses

#     def evaluate(self, x, y):
#         """
#         Evaluate the multi-task LR model on the main task.
#         Returns AUC, precision, recall, F1 for both classes, and BCE loss.
#         """
#         self.eval()
#         with torch.no_grad():
#             main_out, _, _, _ = self.forward(x)
#             preds = (main_out >= 0.5).float().cpu().numpy()
#             y_np = y.cpu().numpy()

#             p, r, f, _ = precision_recall_fscore_support(y_np, preds,
#                                                          labels=[0, 1], zero_division=0)
#             auc = roc_auc_score(y_np, main_out.cpu().numpy())
#             loss = self.main_loss_fn(main_out, y.view(-1)).item()

#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#             "bce_loss": loss
#         }


# class MultiTaskNN(nn.Module):
#     """
#     Multi-task Neural Network with configurable intermediate layers.
#     Uses a shared representation and weighted skip connections for auxiliary tasks.
#     """

#     def __init__(self, input_dim, hidden_dim=16, num_layers=1,
#                  optimizer_type="adam", lr=0.01, lambda_aux=0.3, epochs=300):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             hidden_dim (int): Hidden dimension of each layer.
#             num_layers (int): Number of intermediate fully connected layers.
#             optimizer_type (str): "adam" or "sgd" optimizer.
#             lr (float): Learning rate.
#             lambda_aux (float): Scaling factor for auxiliary losses.
#             epochs (int): Number of training epochs.
#         """
#         super(MultiTaskNN, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.epochs = epochs
#         self.lambda_aux = lambda_aux

#         # Build intermediate feature extractor layers
#         self.feature_layers = nn.ModuleList()
#         for i in range(num_layers):
#             if i == 0:
#                 self.feature_layers.append(nn.Sequential(
#                     nn.Linear(input_dim, hidden_dim),
#                     nn.ReLU()
#                 ))
#             else:
#                 self.feature_layers.append(nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim),
#                     nn.ReLU()
#                 ))

#         # Shared layer (after last intermediate)
#         self.shared = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )

#         # Main task head (binary classification)
#         self.main_task = nn.Sequential(
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )

#         # Auxiliary task heads
#         self.aux_task1 = nn.Linear(hidden_dim, 1)                # regression
#         self.aux_task2 = nn.Linear(hidden_dim, 1)                # regression
#         self.aux_task3 = nn.Sequential(nn.Linear(hidden_dim, 1),  # binary
#                                        nn.Sigmoid())

#         # Learnable α parameters (one per intermediary layer) for skip connections
#         self.alpha_aux1 = nn.Parameter(torch.ones(num_layers))
#         self.alpha_aux2 = nn.Parameter(torch.ones(num_layers))
#         self.alpha_aux3 = nn.Parameter(torch.ones(num_layers))

#         # Optimizer
#         if optimizer_type.lower() == "adam":
#             self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         else:
#             self.optimizer = optim.SGD(self.parameters(), lr=lr)

#         # Loss functions
#         self.main_loss_fn = nn.BCELoss()       # Main binary classification
#         self.aux_loss_fn = nn.MSELoss()        # Aux1 & Aux2 regression
#         self.aux_task3_loss_fn = nn.BCELoss()  # Aux3 binary classification

#     def forward(self, x):
#         """
#         Forward pass:
#           - Pass input through all intermediate layers, storing each output.
#           - Compute main task head on last layer's output.
#           - Compute each auxiliary head by weighted sum over intermediate outputs.
#         Returns:
#             main_out, aux1_out, aux2_out, aux3_out (each of shape [batch_size]).
#         """
#         intermediate_outputs = []
#         h = x
#         for layer in self.feature_layers:
#             h = layer(h)
#             intermediate_outputs.append(h)

#         # Main task: process the last intermediate output
#         h_main = self.shared(intermediate_outputs[-1])
#         main_out = self.main_task(h_main).view(-1)

#         # Weighted-sum helper for auxiliary tasks
#         def weighted_sum(alpha, outputs):
#             weights = torch.softmax(alpha, dim=0)
#             return sum(w * out for w, out in zip(weights, outputs))

#         # Compute auxiliary representations
#         h_aux1 = self.shared(weighted_sum(self.alpha_aux1, intermediate_outputs))
#         h_aux2 = self.shared(weighted_sum(self.alpha_aux2, intermediate_outputs))
#         h_aux3 = self.shared(weighted_sum(self.alpha_aux3, intermediate_outputs))

#         # Auxiliary outputs
#         aux1_out = self.aux_task1(h_aux1).view(-1)
#         aux2_out = self.aux_task2(h_aux2).view(-1)
#         aux3_out = self.aux_task3(h_aux3).view(-1)

#         return main_out, aux1_out, aux2_out, aux3_out

#     def train_model(self, x, main_y, aux1_y=None, aux2_y=None, aux3_y=None,
#                     x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
#                     early_stopping_patience=30, record_loss=False):
#         """
#         Train method for the multi-task neural network.
#         Records only main-task BCE loss for train/validation if record_loss=True.
#         Uses early stopping on the main task validation loss.

#         Args:
#             x (Tensor): Training features.
#             main_y (Tensor): Main binary labels.
#             aux1_y, aux2_y (Tensor): Aux regression targets.
#             aux3_y (Tensor): Aux binary target.
#             x_val (Tensor, optional): Validation features.
#             main_y_val (Tensor, optional): Validation main labels.
#             aux1_val, aux2_val, aux3_val: Validation aux targets.
#             early_stopping_patience (int): Patience on main val loss.
#             record_loss (bool): If True, return (train_loss_history, val_loss_history).

#         Returns:
#             If record_loss=True: train_loss_history, val_loss_history lists.
#         """
#         train_loss_history = []
#         val_loss_history = []
#         best_val_loss = float('inf')
#         patience_counter = 0

#         for epoch in range(self.epochs):
#             self.train()
#             self.optimizer.zero_grad()
#             main_out, a1_out, a2_out, a3_out = self.forward(x)

#             # Main task loss
#             loss_main = self.main_loss_fn(main_out, main_y.view(-1))

#             # Total loss = main + λ * (aux losses)
#             if aux1_y is not None:
#                 loss_aux1 = self.aux_loss_fn(a1_out, aux1_y.view(-1))
#                 loss_aux2 = self.aux_loss_fn(a2_out, aux2_y.view(-1))
#                 loss_aux3 = self.aux_task3_loss_fn(a3_out, aux3_y.view(-1))
#                 total_loss = loss_main + self.lambda_aux * (loss_aux1 + loss_aux2 + loss_aux3)
#             else:
#                 total_loss = loss_main

#             total_loss.backward()
#             self.optimizer.step()

#             if record_loss:
#                 train_loss_history.append(loss_main.item())

#             # Validation step (main task only)
#             if x_val is not None and main_y_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     main_out_val, _, _, _ = self.forward(x_val)
#                     val_loss_main = self.main_loss_fn(main_out_val, main_y_val.view(-1))
#                 if record_loss:
#                     val_loss_history.append(val_loss_main.item())

#                 # Early stopping on main validation loss
#                 if val_loss_main.item() < best_val_loss:
#                     best_val_loss = val_loss_main.item()
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= early_stopping_patience:
#                         break

#         if record_loss:
#             return train_loss_history, val_loss_history

#     def evaluate(self, x, y):
#         """
#         Evaluate the model on the main task only.
#         Returns AUC, per-class precision/recall/F1, and main BCE loss.
#         """
#         self.eval()
#         with torch.no_grad():
#             main_out, _, _, _ = self.forward(x)
#             preds = (main_out >= 0.5).float().cpu().numpy()
#             y_np = y.cpu().numpy()

#             p, r, f, _ = precision_recall_fscore_support(
#                 y_np, preds, labels=[0, 1], zero_division=0
#             )
#             auc = roc_auc_score(y_np, main_out.cpu().numpy())
#             bce_loss = self.main_loss_fn(main_out, y.view(-1)).item()

#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#             "bce_loss": bce_loss
#         }


# class MultiTaskNN_PretrainFinetuneExtended(nn.Module):
#     """
#     Two-phase multi-task network:
#       - Phase 1: Pre-train on auxiliary losses only.
#       - Phase 2: Fine-tune on main task (with optional validation and early stopping).
#       Records pretrain, finetune, and validation losses if requested.
#     """

#     def __init__(self, input_dim, hidden_dim=16, num_layers=1,
#                  lr_pre=0.01, lr_fine=0.005, lambda_aux=0.3,
#                  pre_epochs=100, fine_epochs=100):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             hidden_dim (int): Hidden dimension for feature layers.
#             num_layers (int): Number of intermediate layers.
#             lr_pre (float): Learning rate for pre-training.
#             lr_fine (float): Learning rate for fine-tuning.
#             lambda_aux (float): Weight for auxiliary losses.
#             pre_epochs (int): Number of pre-training epochs.
#             fine_epochs (int): Number of fine-tuning epochs.
#         """
#         super(MultiTaskNN_PretrainFinetuneExtended, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.lambda_aux = lambda_aux
#         self.pre_epochs = pre_epochs
#         self.fine_epochs = fine_epochs

#         # Build feature extractor layers
#         layers = []
#         in_dim = input_dim
#         for _ in range(num_layers):
#             layers.append(nn.Linear(in_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             in_dim = hidden_dim
#         self.feature = nn.Sequential(*layers)

#         # Shared layer after feature extractor
#         self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

#         # Heads
#         self.main = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
#         self.aux1 = nn.Linear(hidden_dim, 1)
#         self.aux2 = nn.Linear(hidden_dim, 1)
#         self.aux3 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

#         # Attention parameters (learnable α) for each intermediate layer output
#         self.alpha1 = nn.Parameter(torch.ones(num_layers))
#         self.alpha2 = nn.Parameter(torch.ones(num_layers))
#         self.alpha3 = nn.Parameter(torch.ones(num_layers))

#         # Loss functions
#         self.main_loss_fn = nn.BCELoss()
#         self.aux_mse = nn.MSELoss()
#         self.aux_bce = nn.BCELoss()

#         # Learning rates
#         self.lr_pre = lr_pre
#         self.lr_fine = lr_fine

#     def forward(self, x):
#         """
#         Forward pass:
#           - Collect outputs from each linear layer for skip connections.
#           - Compute main head on last intermediate output.
#           - Compute auxiliary heads on weighted sums of intermediate outputs.
#         Returns:
#             main_out, aux1_out, aux2_out, aux3_out
#         """
#         outs = []
#         h = x
#         # Pass through feature extractor and record each Linear output
#         for layer in self.feature:
#             h = layer(h)
#             if isinstance(layer, nn.Linear):
#                 outs.append(h)

#         # Main head on last layer output
#         shared_main = self.shared(outs[-1])
#         main_out = self.main(shared_main).view(-1)

#         # Weighted sum helper
#         def weighted(alpha, seq):
#             w = torch.softmax(alpha, dim=0)
#             return sum(w[i] * seq[i] for i in range(len(seq)))

#         s1 = self.shared(weighted(self.alpha1, outs))
#         s2 = self.shared(weighted(self.alpha2, outs))
#         s3 = self.shared(weighted(self.alpha3, outs))

#         a1 = self.aux1(s1).view(-1)
#         a2 = self.aux2(s2).view(-1)
#         a3 = self.aux3(s3).view(-1)

#         return main_out, a1, a2, a3

#     def train_pre_fine(self,
#                       x_pre, y_main_pre, aux1_pre, aux2_pre, aux3_pre,
#                       x_fine, y_fine,
#                       record_loss=False, early_stopping_patience=30,
#                       x_fine_val=None, y_fine_val=None):
#         """
#         Two-phase training:
#           Phase 1: Pre-train on auxiliary tasks using x_pre and aux targets.
#           Phase 2: Fine-tune on main task using x_fine and main labels, with optional validation.
        
#         Args:
#             x_pre (Tensor): Features for pre-training.
#             y_main_pre (Tensor): Main labels for pre-training (unused but kept for API consistency).
#             aux1_pre, aux2_pre, aux3_pre: Auxiliary targets for pre-training.
#             x_fine (Tensor): Features for fine-tuning on main task.
#             y_fine (Tensor): Main labels for fine-tuning.
#             record_loss (bool): If True, returns loss histories.
#             early_stopping_patience (int): Patience for main-task early stopping.
#             x_fine_val (Tensor): Validation features for main task.
#             y_fine_val (Tensor): Validation labels for main task.
        
#         Returns:
#             If record_loss=True: (pre_losses, fine_losses, val_losses).
#         """
#         pre_losses = []
#         fine_losses = []
#         val_losses = []

#         # --- Phase 1: Pre-training on auxiliary tasks ---
#         opt_pre = optim.Adam(
#             list(self.feature.parameters()) +
#             list(self.aux1.parameters()) +
#             list(self.aux2.parameters()) +
#             list(self.aux3.parameters()) +
#             [self.alpha1, self.alpha2, self.alpha3],
#             lr=self.lr_pre
#         )

#         for _ in range(self.pre_epochs):
#             self.train()
#             opt_pre.zero_grad()
#             _, o1, o2, o3 = self.forward(x_pre)
#             l1 = self.aux_mse(o1, aux1_pre.view(-1))
#             l2 = self.aux_mse(o2, aux2_pre.view(-1))
#             l3 = self.aux_bce(o3, aux3_pre.view(-1))
#             loss_side = self.lambda_aux * (l1 + l2 + l3)
#             loss_side.backward()
#             opt_pre.step()
#             if record_loss:
#                 pre_losses.append(loss_side.item())

#         # Freeze auxiliary components
#         self.alpha1.requires_grad = False
#         self.alpha2.requires_grad = False
#         self.alpha3.requires_grad = False
#         for p in self.aux1.parameters(): p.requires_grad = False
#         for p in self.aux2.parameters(): p.requires_grad = False
#         for p in self.aux3.parameters(): p.requires_grad = False

#         # --- Phase 2: Fine-tuning on main task ---
#         opt_fine = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr_fine)
#         best_val = float('inf')
#         patience = 0

#         for epoch in range(self.fine_epochs):
#             self.train()
#             opt_fine.zero_grad()
#             out_main, _, _, _ = self.forward(x_fine)
#             loss_main = self.main_loss_fn(out_main, y_fine.view(-1))
#             loss_main.backward()
#             opt_fine.step()
#             if record_loss:
#                 fine_losses.append(loss_main.item())

#             # Validation step on main task
#             if x_fine_val is not None and y_fine_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     v_main, _, _, _ = self.forward(x_fine_val)
#                     v_loss = self.main_loss_fn(v_main, y_fine_val.view(-1))
#                 if record_loss:
#                     val_losses.append(v_loss.item())
#                 if v_loss.item() < best_val:
#                     best_val = v_loss.item()
#                     patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping_patience:
#                         break

#         if record_loss:
#             return pre_losses, fine_losses, val_losses

#     def train_model(
#         self,
#         x, main_y,
#         aux1=None, aux2=None, aux3=None,
#         x_val=None, main_y_val=None,
#         aux1_val=None, aux2_val=None, aux3_val=None,
#         early_stopping_patience=30, record_loss=False
#     ):
#         """
#         Wrapper that calls train_pre_fine for this two-phase model.
#         """
#         return self.train_pre_fine(
#             x_pre=x,
#             y_main_pre=main_y,
#             aux1_pre=aux1,
#             aux2_pre=aux2,
#             aux3_pre=aux3,
#             x_fine=x,
#             y_fine=main_y,
#             record_loss=record_loss,
#             early_stopping_patience=early_stopping_patience,
#             x_fine_val=x_val,
#             y_fine_val=main_y_val
#         )

#     def evaluate(self, x, y):
#         """
#         Evaluate the model on the main task.
#         Returns AUC, per-class precision/recall/F1, and BCE loss.
#         """
#         self.eval()
#         with torch.no_grad():
#             out_main, _, _, _ = self.forward(x)
#             preds = (out_main >= 0.5).float().cpu().numpy()
#             y_np = y.cpu().numpy()

#             p, r, f, _ = precision_recall_fscore_support(
#                 y_np, preds, labels=[0, 1], zero_division=0
#             )
#             auc = roc_auc_score(y_np, out_main.cpu().numpy())
#             loss = self.main_loss_fn(out_main, y.view(-1)).item()

#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#             "bce_loss": loss
#         }


# class MultiTaskNN_Decoupled(nn.Module):
#     """
#     Decoupled multi-task network:
#       - Phase 1: Optimize auxiliary losses for φ & β.
#       - Phase 2: Freeze φ & β, then optimize main head ψ with optional validation.
#     """

#     def __init__(self, input_dim, hidden_dim=16, num_layers=1,
#                  lr=0.01, lambda_aux=0.3,
#                  pre_epochs=100, main_epochs=100):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             hidden_dim (int): Hidden dimension for feature layers.
#             num_layers (int): Number of intermediate layers.
#             lr (float): Learning rate for both phases.
#             lambda_aux (float): Weight for auxiliary losses.
#             pre_epochs (int): Number of epochs for auxiliary optimization.
#             main_epochs (int): Number of epochs for main-task optimization.
#         """
#         super(MultiTaskNN_Decoupled, self).__init__()
#         self.lambda_aux = lambda_aux
#         self.pre_epochs = pre_epochs
#         self.main_epochs = main_epochs
#         self.lr = lr

#         # Build feature extractor
#         layers = []
#         in_dim = input_dim
#         for _ in range(num_layers):
#             layers.append(nn.Linear(in_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             in_dim = hidden_dim
#         self.feature = nn.Sequential(*layers)

#         # Shared layer
#         self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

#         # Heads
#         self.main = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
#         self.aux1 = nn.Linear(hidden_dim, 1)
#         self.aux2 = nn.Linear(hidden_dim, 1)
#         self.aux3 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

#         # Attention α parameters
#         self.alpha1 = nn.Parameter(torch.ones(num_layers))
#         self.alpha2 = nn.Parameter(torch.ones(num_layers))
#         self.alpha3 = nn.Parameter(torch.ones(num_layers))

#         # Loss functions
#         self.main_loss_fn = nn.BCELoss()
#         self.aux_mse = nn.MSELoss()
#         self.aux_bce = nn.BCELoss()

#     def forward(self, x):
#         """
#         Forward pass:
#           - Collect intermediate outputs from feature extractor.
#           - Compute main head on last intermediate output.
#           - Compute auxiliary heads on weighted sums of intermediate outputs.
#         Returns:
#             main_out, aux1_out, aux2_out, aux3_out
#         """
#         outs = []
#         h = x
#         for layer in self.feature:
#             h = layer(h)
#             if isinstance(layer, nn.Linear):
#                 outs.append(h)

#         shared_main = self.shared(outs[-1])
#         main_out = self.main(shared_main).view(-1)

#         def weighted(alpha, seq):
#             w = torch.softmax(alpha, dim=0)
#             return sum(w[i] * seq[i] for i in range(len(seq)))

#         s1 = self.shared(weighted(self.alpha1, outs))
#         s2 = self.shared(weighted(self.alpha2, outs))
#         s3 = self.shared(weighted(self.alpha3, outs))

#         a1 = self.aux1(s1).view(-1)
#         a2 = self.aux2(s2).view(-1)
#         a3 = self.aux3(s3).view(-1)

#         return main_out, a1, a2, a3

#     def train_model(
#         self,
#         x, y_main, aux1_y, aux2_y, aux3_y,
#         x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
#         early_stopping_patience=30,
#         record_loss=False
#     ):
#         """
#         Two-phase training:
#           Phase 1: Optimize auxiliary losses (φ & β) on all training data.
#           Phase 2: Freeze φ & β, then optimize main head ψ on the same data or validation.

#         Args:
#             x (Tensor): Features for both phases.
#             y_main (Tensor): Main labels.
#             aux1_y, aux2_y, aux3_y (Tensor): Auxiliary targets.
#             x_val, main_y_val (Tensor): Validation for main phase.
#             early_stopping_patience (int): Patience for early stopping on main task.
#             record_loss (bool): If True, print or return loss histories.
#         Returns:
#             If record_loss=True: prints main_losses and pre_losses.
#         """
#         pre_losses = []
#         main_losses = []
#         val_losses = []

#         # --- Phase 1: Auxiliary Optimization ---
#         opt_pre = optim.Adam(
#             list(self.feature.parameters()) +
#             list(self.aux1.parameters()) +
#             list(self.aux2.parameters()) +
#             list(self.aux3.parameters()) +
#             [self.alpha1, self.alpha2, self.alpha3],
#             lr=self.lr
#         )
#         for _ in range(self.pre_epochs):
#             self.train()
#             opt_pre.zero_grad()
#             _, o1, o2, o3 = self.forward(x)
#             loss_side = self.lambda_aux * (
#                 self.aux_mse(o1, aux1_y.view(-1)) +
#                 self.aux_mse(o2, aux2_y.view(-1)) +
#                 self.aux_bce(o3, aux3_y.view(-1))
#             )
#             loss_side.backward()
#             opt_pre.step()
#             if record_loss:
#                 pre_losses.append(loss_side.item())

#         # Freeze φ & β (feature extractor, shared, and aux heads)
#         for p in self.feature.parameters():
#             p.requires_grad = False
#         for p in self.shared.parameters():
#             p.requires_grad = False
#         for p in self.aux1.parameters():
#             p.requires_grad = False
#         for p in self.aux2.parameters():
#             p.requires_grad = False
#         for p in self.aux3.parameters():
#             p.requires_grad = False
#         self.alpha1.requires_grad = False
#         self.alpha2.requires_grad = False
#         self.alpha3.requires_grad = False

#         # --- Phase 2: Main Task Optimization with Validation ---
#         opt_main = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
#         best_val = float('inf')
#         patience = 0

#         for _ in range(self.main_epochs):
#             self.train()
#             opt_main.zero_grad()
#             out_main, _, _, _ = self.forward(x)
#             loss_main = self.main_loss_fn(out_main, y_main.view(-1))
#             loss_main.backward()
#             opt_main.step()
#             if record_loss:
#                 main_losses.append(loss_main.item())

#             if x_val is not None and main_y_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     v_main, _, _, _ = self.forward(x_val)
#                     v_loss = self.main_loss_fn(v_main, main_y_val.view(-1))
#                 if record_loss:
#                     val_losses.append(v_loss.item())
#                 if v_loss.item() < best_val:
#                     best_val = v_loss.item()
#                     patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping_patience:
#                         break
#             self.train()

#         if record_loss:
#             print("Main losses:", main_losses)
#             print("Pre-training losses:", pre_losses)
#             return main_losses, val_losses

#     def evaluate(self, x, y):
#         """
#         Evaluate the model on the main task.
#         Returns AUC, precision/recall/F1 for each class, and BCE loss.
#         """
#         self.eval()
#         with torch.no_grad():
#             out_main, _, _, _ = self.forward(x)
#             preds = (out_main >= 0.5).float().cpu().numpy()
#             y_np = y.cpu().numpy()

#             p, r, f, _ = precision_recall_fscore_support(y_np, preds,
#                                                          labels=[0, 1], zero_division=0)
#             auc = roc_auc_score(y_np, out_main.cpu().numpy())
#             loss = self.main_loss_fn(out_main, y.view(-1)).item()

#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#             "bce_loss": loss
#         }


# # =========================
# #       GRID SEARCH
# # =========================

# def grid_search_cv(model_class, param_grid, X, y, cv=3, is_multitask=False,
#                    aux1=None, aux2=None, aux3=None):
#     """
#     Perform grid search using ROC AUC on validation for main task.
#     For multi-task models, only the main task validation AUC is used.

#     Args:
#         model_class (class): The model class to instantiate (e.g. MultiTaskNN).
#         param_grid (dict): Dictionary of hyperparameter lists.
#         X (Tensor): Features for grid-search (balanced set).
#         y (Tensor): Main binary labels for grid-search.
#         cv (int): Number of cross-validation folds (default: 3).
#         is_multitask (bool): If True, uses aux1, aux2, aux3 during training.
#         aux1, aux2, aux3 (Tensor): Auxiliary targets for training/validation.

#     Returns:
#         best_params (dict): Hyperparameter combination with highest average AUC.
#         best_auc (float): Corresponding average validation AUC.
#     """
#     best_auc = -1.0
#     best_params = None
#     keys = list(param_grid.keys())

#     X_np = X.numpy() if hasattr(X, 'numpy') else X
#     y_np = y.numpy() if hasattr(y, 'numpy') else y

#     for combo in itertools.product(*(param_grid[k] for k in keys)):
#         params = dict(zip(keys, combo))
#         aucs = []
#         kf = KFold(n_splits=cv, shuffle=True, random_state=42)

#         for train_idx, val_idx in kf.split(X_np):
#             x_tr = torch.tensor(X_np[train_idx], dtype=torch.float32)
#             x_val = torch.tensor(X_np[val_idx], dtype=torch.float32)
#             y_tr = torch.tensor(y_np[train_idx], dtype=torch.float32)
#             y_val = torch.tensor(y_np[val_idx], dtype=torch.float32)

#             if is_multitask:
#                 aux1_tr = aux1[train_idx]
#                 aux2_tr = aux2[train_idx]
#                 aux3_tr = aux3[train_idx]
#             else:
#                 aux1_tr = aux2_tr = aux3_tr = None

#             sig = inspect.signature(model_class.__init__).parameters
#             # Instantiate model with input_dim if needed
#             if 'input_dim' in sig:
#                 model = model_class(input_dim=X_np.shape[1], **params)
#             else:
#                 model = model_class(**params)

#             if is_multitask:
#                 model.train_model(
#                     x_tr, y_tr, aux1_tr, aux2_tr, aux3_tr,
#                     x_val=x_val, main_y_val=y_val,
#                     early_stopping_patience=10,
#                     record_loss=False
#                 )
#             else:
#                 model.train_model(
#                     x_tr, y_tr,
#                     x_val=x_val, main_y_val=y_val,
#                     early_stopping_patience=10,
#                     record_loss=False
#                 )

#             with torch.no_grad():
#                 out_main = model.forward(x_val)[0]
#                 auc_score = roc_auc_score(y_val.cpu().numpy(), out_main.cpu().numpy())
#                 aucs.append(auc_score)

#         avg_auc = float(torch.tensor(aucs).mean().item())
#         if avg_auc > best_auc:
#             best_auc = avg_auc
#             best_params = params.copy()

#     return best_params, best_auc


# # =========================
# #   BACKWARD COMPATIBILITY
# # =========================

# def grid_search_cv_basline(model_class, param_grid, X, y, cv=3,
#                            aux1=None, aux2=None, aux3=None):
#     """
#     Legacy grid search (same as above) for backward compatibility.
#     """
#     return grid_search_cv(model_class, param_grid, X, y, cv,
#                           is_multitask=(aux1 is not None),
#                           aux1=aux1, aux2=aux2, aux3=aux3)


# # =========================
# #   RUN EXPERIMENTS FUNCTION
# # =========================

# def run_experiments(model_class, method_name, num_runs, best_params,
#                     train_x_pool, train_y_pool, test_x_pool, test_y_pool,
#                     optimizer_fixed="adam", epochs_fixed=300,
#                     is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
#     """
#     Run multiple experiments with sampling and evaluation.
#     For each run:
#       1. Sample 250 positives + 250 negatives from train pool → 500 train samples.
#       2. Sample 125 test samples from test pool.
#       3. Initialize the model based on best_params.
#       4. Split the 500 train samples into train/val (80/20).
#       5. Train the model with optional auxiliary targets.
#       6. Plot train vs. validation main-task loss.
#       7. Evaluate on the 125 test samples and record metrics.

#     Args:
#         model_class (class): Model class to instantiate (string check for different subclasses).
#         method_name (str): Name for saving plots, logging.
#         num_runs (int): Number of runs to average over.
#         best_params (dict): Hyperparameters determined by grid search.
#         train_x_pool (Tensor): Pool of training features.
#         train_y_pool (Tensor): Pool of main training labels.
#         test_x_pool (Tensor): Pool of test features.
#         test_y_pool (Tensor): Pool of test labels.
#         optimizer_fixed (str): Optimizer type ("adam" or "sgd").
#         epochs_fixed (int): Number of epochs if not specified in best_params.
#         is_multitask (bool): If True, expects auxiliary pools to be provided.
#         train_aux1_pool (Tensor): Pool of aux1 targets (health_1_10).
#         train_aux2_pool (Tensor): Pool of aux2 targets (diabetes_risk_score).
#         train_aux3_pool (Tensor): Pool of aux3 targets (has_diabetes).

#     Returns:
#         metrics_list (List[dict]): List of evaluation metrics for each run.
#     """
#     metrics_list = []
#     class_counts = []

#     # Build a DataFrame to extract class indices for balanced sampling
#     train_df = pd.DataFrame(train_x_pool.numpy())
#     train_df['Diabetes_binary'] = train_y_pool.numpy()
#     class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
#     class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
#     test_pool_size = len(test_x_pool)

#     # Extract hyperparameters with default fallbacks
#     lr_val = best_params.get('lr', 0.01)
#     hidden_dim_val = best_params.get('hidden_dim', 64)
#     num_layers_val = best_params.get('num_layers', 1)
#     lambda_aux_val = best_params.get('lambda_aux', 0.3)

#     for run_idx in range(num_runs):
#         # ----------------------------
#         # 1) Balanced sampling: 250 pos, 250 neg
#         # ----------------------------
#         sampled1 = random.sample(class1_indices, 250)
#         sampled0 = random.sample(class0_indices, 250)
#         train_idx = sampled1 + sampled0
#         x_train_run = train_x_pool[train_idx]
#         y_train_run = train_y_pool[train_idx]
#         n_pos = int((y_train_run == 1).sum().item())
#         n_neg = int((y_train_run == 0).sum().item())
#         class_counts.append({'run': run_idx + 1, 'n_neg': n_neg, 'n_pos': n_pos})
#         print(f"Run {run_idx + 1}: negative={n_neg}, positive={n_pos}")

#         if is_multitask:
#             aux1_train_run = train_aux1_pool[train_idx]
#             aux2_train_run = train_aux2_pool[train_idx]
#             aux3_train_run = train_aux3_pool[train_idx]
#         else:
#             aux1_train_run = aux2_train_run = aux3_train_run = None

#         # ----------------------------
#         # 2) Randomly select 125 test samples
#         # ----------------------------
#         test_indices = random.sample(range(test_pool_size), 125)
#         x_test_run = test_x_pool[test_indices]
#         y_test_run = test_y_pool[test_indices]

#         # ----------------------------
#         # 3) Instantiate model based on class name and best_params
#         # ----------------------------
#         model_name = model_class.__name__
#         input_dim = x_train_run.shape[1]

#         if model_name == "MultiTaskLogisticRegression":
#             model = model_class(
#                 input_dim=input_dim,
#                 lr=lr_val,
#                 optimizer_type=optimizer_fixed,
#                 lambda_aux1=best_params.get('lambda_aux1', 0.3),
#                 lambda_aux2=best_params.get('lambda_aux2', 0.3),
#                 lambda_aux3=best_params.get('lambda_aux3', 0.3),
#                 epochs=epochs_fixed
#             )
#         elif model_name == "MultiTaskNN_PretrainFinetuneExtended":
#             model = model_class(
#                 input_dim=input_dim,
#                 hidden_dim=best_params['hidden_dim'],
#                 num_layers=best_params['num_layers'],
#                 lambda_aux=best_params['lambda_aux'],
#                 lr_pre=best_params['lr_pre'],
#                 lr_fine=best_params['lr_fine'],
#                 pre_epochs=best_params['pre_epochs'],
#                 fine_epochs=best_params['fine_epochs']
#             )
#         elif model_name == "MultiTaskNN_Decoupled":
#             model = model_class(
#                 input_dim=input_dim,
#                 hidden_dim=best_params['hidden_dim'],
#                 num_layers=best_params['num_layers'],
#                 lambda_aux=best_params['lambda_aux'],
#                 lr=best_params['lr'],
#                 pre_epochs=best_params.get('pre_epochs', 100),
#                 main_epochs=best_params.get('fine_epochs', 100)
#             )
#         else:  # Default: MultiTaskNN
#             model = model_class(
#                 input_dim=input_dim,
#                 hidden_dim=hidden_dim_val,
#                 num_layers=num_layers_val,
#                 optimizer_type=optimizer_fixed,
#                 lr=lr_val,
#                 lambda_aux=lambda_aux_val,
#                 epochs=epochs_fixed
#             )

#         # ----------------------------
#         # 4) Split train_run into train/val (80/20)
#         # ----------------------------
#         indices = list(range(len(x_train_run)))
#         random.shuffle(indices)
#         split = int(0.8 * len(indices))
#         train_sub_idx, val_sub_idx = indices[:split], indices[split:]

#         x_train_sub = x_train_run[train_sub_idx]
#         y_train_sub = y_train_run[train_sub_idx]
#         x_val_sub = x_train_run[val_sub_idx]
#         y_val_sub = y_train_run[val_sub_idx]

#         if is_multitask:
#             aux1_train_sub = aux1_train_run[train_sub_idx]
#             aux2_train_sub = aux2_train_run[train_sub_idx]
#             aux3_train_sub = aux3_train_run[train_sub_idx]
#             aux1_val_sub = aux1_train_run[val_sub_idx]
#             aux2_val_sub = aux2_train_run[val_sub_idx]
#             aux3_val_sub = aux3_train_run[val_sub_idx]
#         else:
#             aux1_train_sub = aux2_train_sub = aux3_train_sub = None
#             aux1_val_sub = aux2_val_sub = aux3_val_sub = None

#         # ----------------------------
#         # 5) Train the model
#         # ----------------------------
#         if model_name == "MultiTaskNN_PretrainFinetuneExtended":
#             # This returns (pre_losses, fine_losses, val_losses)
#             pre_losses, train_losses, val_losses = model.train_model(
#                 x_pre=x_train_sub,
#                 y_main_pre=y_train_sub,
#                 aux1_pre=aux1_train_sub,
#                 aux2_pre=aux2_train_sub,
#                 aux3_pre=aux3_train_sub,
#                 x_fine=x_train_sub,
#                 y_fine=y_train_sub,
#                 x_fine_val=x_val_sub,
#                 y_fine_val=y_val_sub,
#                 record_loss=True,
#                 early_stopping_patience=30
#             )
#         else:
#             train_val_losses = model.train_model(
#                 x=x_train_sub,
#                 main_y=y_train_sub,
#                 aux1_y=aux1_train_sub,
#                 aux2_y=aux2_train_sub,
#                 aux3_y=aux3_train_sub,
#                 x_val=x_val_sub,
#                 main_y_val=y_val_sub,
#                 aux1_val=aux1_val_sub,
#                 aux2_val=aux2_val_sub,
#                 aux3_val=aux3_val_sub,
#                 early_stopping_patience=30,
#                 record_loss=True
#             )
#             if train_val_losses is not None:
#                 train_losses, val_losses = train_val_losses
#             else:
#                 train_losses, val_losses = [], []

#         # ----------------------------
#         # 6) Plot train vs. validation main-task loss
#         # ----------------------------
#         if not os.path.exists("img"):
#             os.makedirs("img")
#         plot_filename = os.path.join("img", f"{method_name}_run{run_idx+1}.png")
#         plt.figure(figsize=(8, 6))
#         if train_losses:
#             plt.plot(train_losses, label="Train Main BCE Loss")
#         if val_losses:
#             plt.plot(val_losses, label="Validation Main BCE Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.title(f"{method_name} Run {run_idx+1}: Main Task Loss")
#         plt.legend()
#         plt.savefig(plot_filename)
#         plt.close()

#         # ----------------------------
#         # 7) Evaluate on test set
#         # ----------------------------
#         metrics = model.evaluate(x_test_run, y_test_run)
#         metrics_list.append(metrics)

#     return metrics_list


# # =========================
# #   AGGREGATION & UTILITIES
# # =========================

# def aggregate_metrics(metrics_list):
#     """
#     Aggregate a list of metric dictionaries by computing mean ± std for each metric key.
#     """
#     agg = {}
#     for key in metrics_list[0].keys():
#         vals = [m[key] for m in metrics_list]
#         mean_val = float(torch.tensor(vals).mean().item())
#         std_val = float(torch.tensor(vals).std().item())
#         agg[key] = f"{mean_val:.4f} ± {std_val:.4f}"
#     return agg


# def write_results_csv(filename, method, metrics, params):
#     """
#     Append a row of results (metrics + params) to a CSV file.
#     """
#     with open(filename, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             method,
#             metrics.get('auc', ''),
#             metrics.get('p0', ''), metrics.get('r0', ''), metrics.get('f0', ''),
#             metrics.get('p1', ''), metrics.get('r1', ''), metrics.get('f1', ''),
#             metrics.get('bce_loss', ''),
#             params
#         ])


# def print_aggregated_results(method, metrics, params):
#     """
#     Print aggregated results in a readable format.
#     """
#     print("\n=== Final Aggregated Results ===")
#     print(f"Method: {method}")
#     print(f"AUC: {metrics.get('auc','')}")
#     print(f"Class0 → Precision: {metrics.get('p0','')}, Recall: {metrics.get('r0','')}, F1: {metrics.get('f0','')}")
#     print(f"Class1 → Precision: {metrics.get('p1','')}, Recall: {metrics.get('r1','')}, F1: {metrics.get('f1','')}")
#     print(f"Main BCE Loss: {metrics.get('bce_loss','')}")
#     print(f"Best Parameters: {params}")
#     print("---------------------------------------------------")


# # =========================
# #          MAIN
# # =========================

# def main():
#     """
#     Main entry point to run multi-task experiments.
#     Loads data, performs grid search, runs experiments, and saves results.
#     """
#     # 1. Ensure reproducibility
#     set_seed(SEED)

#     OUTPUT_CSV = "multitask_results.csv"
#     if not os.path.exists(OUTPUT_CSV):
#         with open(OUTPUT_CSV, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "Method", "AUC",
#                 "Acc0", "Prec0", "Rec0", "F10",
#                 "Acc1", "Prec1", "Rec1", "F11",
#                 "BCE_Loss", "Params"
#             ])

#     # 2. Preprocess data
#     dp = MultiTaskDatasetPreprocessor(
#         dataset_id=891,
#         side_info_path='prompting/augmented_data_70B.csv'
#     )
#     (grid_x, grid_y, grid_a1, grid_a2, grid_a3,
#      train_x, train_y, ta1, ta2, ta3,
#      test_x, test_y) = dp.preprocess()

#     # 3. Define experiments
#     experiments = [
#         {
#             "name": "Decoupled",
#             "model": MultiTaskNN_Decoupled,
#             "param_grid": {
#                 "lr": [0.01, 0.001],
#                 "hidden_dim": [64, 128, 256],
#                 "num_layers": [1, 2, 3, 4],
#                 "lambda_aux": [0.1, 0.3, 0.01]
#             }
#         }
#     ]

#     # 4. Loop over each experiment configuration
#     for cfg in experiments:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         name = cfg["name"]
#         Model = cfg["model"]
#         param_grid = cfg["param_grid"]

#         # 4a. Grid search (use multi-task CV)
#         best_params, best_auc = grid_search_cv(
#             Model, param_grid,
#             grid_x, grid_y,
#             cv=3,
#             is_multitask=True,
#             aux1=grid_a1, aux2=grid_a2, aux3=grid_a3
#         )
#         print(f"[{name}] Best params: {best_params}, AUC: {best_auc:.4f}")

#         # 4b. Final repeated runs using best_params
#         results = run_experiments(
#             Model, name, num_runs=10, best_params=best_params,
#             train_x_pool=train_x, train_y_pool=train_y,
#             test_x_pool=test_x, test_y_pool=test_y,
#             is_multitask=True,
#             train_aux1_pool=ta1, train_aux2_pool=ta2, train_aux3_pool=ta3
#         )

#         # 4c. Aggregate metrics (mean ± std)
#         agg = aggregate_metrics(results)

#         # 4d. Save to CSV and print
#         write_results_csv(OUTPUT_CSV, name, agg, str(best_params))
#         print_aggregated_results(name, agg, str(best_params))


# if __name__ == "__main__":
#     main()


# multitask_project.py

# =========================
#        IMPORTS
# =========================

import os
import csv
import random
import itertools
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, log_loss

from ucimlrepo import fetch_ucirepo


# =========================
#   GLOBAL SEED FUNCTION
# =========================

def set_seed(seed: int):
    """
    Set the random seed for reproducibility across random, NumPy, and PyTorch.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
#   DATA PREPROCESSOR
# =========================

class MultiTaskDatasetPreprocessor:
    """
    Preprocessor for the multi-task model.
    Combines LLM-augmented data with original UCI data.
    Produces training pools and test set with auxiliary targets.
    """

    def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
        """
        Args:
            dataset_id (int): ID of the UCI dataset to fetch (default: 891).
            side_info_path (str): Path to CSV containing LLM-augmented side information.
        """
        # Fetch original UCI dataset
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features  # pandas DataFrame
        self.y_original = self.original_data.data.targets   # pandas Series

        # Standardizer for continuous features
        self.scaler = StandardScaler()

        # Path to side information CSV
        self.side_info_path = side_info_path

        # Columns used as input features
        self.training_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
            'Income'
        ]
        # Columns for main and auxiliary targets
        self.target_cols = ['has_diabetes', 'health_1_10', 'diabetes_risk_score', 'Diabetes_binary']

    def preprocess(self):
        """
        Execute preprocessing and return tensors for:
          - grid set (balanced) for hyperparameter tuning (not used in integrated CV)
          - training pool with auxiliary targets
          - test set (only main task)
        
        Returns:
            tuple: (train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
                    test_x_pool, test_y_pool)
        """
        continuous_cols = ['BMI']

        # 1) Load the augmented dataset with side information
        augmented_df = pd.read_csv(self.side_info_path)
        augmented_df = augmented_df[self.training_cols + self.target_cols]

        # 2) Fill or drop missing values in auxiliary and main targets
        augmented_df['has_diabetes'].fillna(0, inplace=True)
        augmented_df['health_1_10'].fillna(augmented_df['health_1_10'].median(), inplace=True)
        augmented_df['diabetes_risk_score'].fillna(augmented_df['diabetes_risk_score'].median(), inplace=True)
        augmented_df['Diabetes_binary'].fillna(0, inplace=True)

        # 3) Remaining training data used as 'train pool'
        train_pool = augmented_df.copy()

        # 4) Construct a separate test set using original dataset (no augmented info)
        original_df = self.X_original.copy()
        original_df['Diabetes_binary'] = self.y_original
        # Exclude rows that appear in augmented_df to avoid overlap
        test_pool = original_df.drop(index=augmented_df.index, errors='ignore')
        test_pool = test_pool[self.training_cols + ['Diabetes_binary']]

        # 5) Fit scaler on continuous columns of train_pool, then transform train_pool + test_pool
        self.scaler.fit(train_pool[continuous_cols])
        train_pool_scaled = train_pool.copy()
        train_pool_scaled[continuous_cols] = self.scaler.transform(train_pool[continuous_cols])
        test_pool_scaled = test_pool.copy()
        test_pool_scaled[continuous_cols] = self.scaler.transform(test_pool[continuous_cols])

        # 6) Convert train_pool to PyTorch tensors (multi-task)
        def df_to_tensors_multi(df):
            features = df.drop(columns=self.target_cols, errors='ignore')
            x = torch.from_numpy(features.values).float()
            y_main = torch.from_numpy(df['Diabetes_binary'].values).float()
            y_aux1 = torch.from_numpy(df['health_1_10'].values).float()
            y_aux2 = torch.from_numpy(df['diabetes_risk_score'].values).float()
            y_aux3 = torch.from_numpy(df['has_diabetes'].values).float()
            return x, y_main, y_aux1, y_aux2, y_aux3

        train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool = \
            df_to_tensors_multi(train_pool_scaled)

        # 7) Convert test set to tensors (only main task)
        def df_to_tensors_test(df):
            features = df[self.training_cols]
            x = torch.from_numpy(features.values).float()
            y = torch.from_numpy(df['Diabetes_binary'].values).float()
            return x, y

        test_x_pool, test_y_pool = df_to_tensors_test(test_pool_scaled)

        return (
            train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
            test_x_pool, test_y_pool
        )


# =========================
#       MODEL CLASSES
# =========================

SEED = 42  # Global seed for PyTorch reproducibility

class MultiTaskLogisticRegression(nn.Module):
    """
    Multi-task Logistic Regression:
      - Shares a linear layer for feature transformation + batch norm.
      - Main head: binary classification with BCE loss.
      - Aux1 & Aux2: regression tasks with MSE loss.
      - Aux3: binary auxiliary task with BCE loss.
      - Total loss = main_bce + λ1*MSE(aux1) + λ2*MSE(aux2) + λ3*BCE(aux3).
    """

    def __init__(self, input_dim, optimizer_type="adam", lr=0.01,
                 lambda_aux1=0.3, lambda_aux2=0.3, lambda_aux3=0.3,
                 epochs=300):
        """
        Args:
            input_dim (int): Number of input features.
            optimizer_type (str): "adam" or "sgd" for optimizer selection.
            lr (float): Learning rate.
            lambda_aux1 (float): Weight for aux1 (MSE) loss.
            lambda_aux2 (float): Weight for aux2 (MSE) loss.
            lambda_aux3 (float): Weight for aux3 (BCE) loss.
            epochs (int): Number of training epochs.
        """
        super().__init__()
        # Shared linear + batch norm layer
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )

        # Main task head (binary classification)
        self.main_head = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

        # Auxiliary heads
        self.aux1 = nn.Linear(input_dim, 1)                 # regression
        self.aux2 = nn.Linear(input_dim, 1)                 # regression
        self.aux3 = nn.Sequential(nn.Linear(input_dim, 1),  # binary
                                   nn.Sigmoid())

        self.lr = lr
        self.epochs = epochs
        self.lambda_aux1 = lambda_aux1
        self.lambda_aux2 = lambda_aux2
        self.lambda_aux3 = lambda_aux3

        # Choose optimizer
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

        # Loss functions
        self.main_loss_fn = nn.BCELoss()
        self.aux_mse = nn.MSELoss()
        self.aux_bce = nn.BCELoss()

    def forward(self, x):
        """
        Forward pass returning main_out, aux1_out, aux2_out, aux3_out.
        """
        feat = self.shared(x)
        main = self.main_head(feat).view(-1)
        a1 = self.aux1(feat).view(-1)
        a2 = self.aux2(feat).view(-1)
        a3 = self.aux3(feat).view(-1)
        return main, a1, a2, a3

    def train_model(self, x, main_y, aux1_y=None, aux2_y=None, aux3_y=None,
                    x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
                    early_stopping_patience=30, record_loss=False):
        """
        Train the multi-task LR model.
        Args:
            x (Tensor): Training features.
            main_y (Tensor): Main binary labels.
            aux1_y (Tensor): Aux1 regression targets.
            aux2_y (Tensor): Aux2 regression targets.
            aux3_y (Tensor): Aux3 binary targets.
            x_val (Tensor, optional): Validation features.
            main_y_val (Tensor, optional): Validation main labels.
            aux1_val, aux2_val, aux3_val: Validation aux targets.
            early_stopping_patience (int): Patience for early stopping on main validation loss.
            record_loss (bool): If True, returns (train_losses, val_losses) for the main loss.
        Returns:
            If record_loss: (train_losses, val_losses) lists of main-task BCE losses.
        """
        train_losses = []
        val_losses = []
        best_val = float('inf')
        patience = 0

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            main_out, a1_out, a2_out, a3_out = self.forward(x)

            # Compute main loss (BCE)
            loss_main = self.main_loss_fn(main_out, main_y.view(-1))

            # Compute auxiliary losses if provided
            if aux1_y is not None:
                loss_a1 = self.aux_mse(a1_out, aux1_y.view(-1))
                loss_a2 = self.aux_mse(a2_out, aux2_y.view(-1))
                loss_a3 = self.aux_bce(a3_out, aux3_y.view(-1))
                total_loss = (loss_main +
                              self.lambda_aux1 * loss_a1 +
                              self.lambda_aux2 * loss_a2 +
                              self.lambda_aux3 * loss_a3)
            else:
                total_loss = loss_main

            total_loss.backward()
            self.optimizer.step()

            if record_loss:
                train_losses.append(loss_main.item())

            # Validation step (for main task only)
            if x_val is not None and main_y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_main, _, _, _ = self.forward(x_val)
                    v_loss = self.main_loss_fn(val_main, main_y_val.view(-1))
                if record_loss:
                    val_losses.append(v_loss.item())

                # Early stopping on main validation loss
                if v_loss.item() < best_val:
                    best_val = v_loss.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break

        if record_loss:
            return train_losses, val_losses

    def evaluate(self, x, y):
        """
        Evaluate the multi-task LR model on the main task.
        Returns AUC, precision, recall, F1 for both classes, and BCE loss.
        """
        self.eval()
        with torch.no_grad():
            main_out, _, _, _ = self.forward(x)
            preds = (main_out >= 0.5).float().cpu().numpy()
            y_np = y.cpu().numpy()

            p, r, f, _ = precision_recall_fscore_support(y_np, preds,
                                                         labels=[0, 1], zero_division=0)
            auc = roc_auc_score(y_np, main_out.cpu().numpy())
            loss = self.main_loss_fn(main_out, y.view(-1)).item()

        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
            "bce_loss": loss
        }


class MultiTaskNN(nn.Module):
    """
    Multi-task Neural Network with configurable intermediate layers.
    Uses a shared representation and weighted skip connections for auxiliary tasks.
    """

    def __init__(self, input_dim, hidden_dim=16, num_layers=1,
                 optimizer_type="adam", lr=0.01, lambda_aux=0.3, epochs=300):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension of each layer.
            num_layers (int): Number of intermediate fully connected layers.
            optimizer_type (str): "adam" or "sgd" optimizer.
            lr (float): Learning rate.
            lambda_aux (float): Scaling factor for auxiliary losses.
            epochs (int): Number of training epochs.
        """
        super(MultiTaskNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lambda_aux = lambda_aux

        # Build intermediate feature extractor layers
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU()
                ))
            else:
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ))

        # Shared layer (after last intermediate)
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
        self.aux_task1 = nn.Linear(hidden_dim, 1)                # regression
        self.aux_task2 = nn.Linear(hidden_dim, 1)                # regression
        self.aux_task3 = nn.Sequential(nn.Linear(hidden_dim, 1),  # binary
                                       nn.Sigmoid())

        # Learnable α parameters (one per intermediary layer) for skip connections
        self.alpha_aux1 = nn.Parameter(torch.ones(num_layers))
        self.alpha_aux2 = nn.Parameter(torch.ones(num_layers))
        self.alpha_aux3 = nn.Parameter(torch.ones(num_layers))

        # Optimizer
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=lr)

        # Loss functions
        self.main_loss_fn = nn.BCELoss()       # Main binary classification
        self.aux_loss_fn = nn.MSELoss()        # Aux1 & Aux2 regression
        self.aux_task3_loss_fn = nn.BCELoss()  # Aux3 binary classification

    def forward(self, x):
        """
        Forward pass:
          - Pass input through all intermediate layers, storing each output.
          - Compute main task head on last layer's output.
          - Compute each auxiliary head by weighted sum over intermediate outputs.
        Returns:
            main_out, aux1_out, aux2_out, aux3_out (each of shape [batch_size]).
        """
        intermediate_outputs = []
        h = x
        for layer in self.feature_layers:
            h = layer(h)
            intermediate_outputs.append(h)

        # Main task: process the last intermediate output
        h_main = self.shared(intermediate_outputs[-1])
        main_out = self.main_task(h_main).view(-1)

        # Weighted-sum helper for auxiliary tasks
        def weighted_sum(alpha, outputs):
            weights = torch.softmax(alpha, dim=0)
            return sum(w * out for w, out in zip(weights, outputs))

        # Compute auxiliary representations
        h_aux1 = self.shared(weighted_sum(self.alpha_aux1, intermediate_outputs))
        h_aux2 = self.shared(weighted_sum(self.alpha_aux2, intermediate_outputs))
        h_aux3 = self.shared(weighted_sum(self.alpha_aux3, intermediate_outputs))

        # Auxiliary outputs
        aux1_out = self.aux_task1(h_aux1).view(-1)
        aux2_out = self.aux_task2(h_aux2).view(-1)
        aux3_out = self.aux_task3(h_aux3).view(-1)

        return main_out, aux1_out, aux2_out, aux3_out

    def train_model(self, x, main_y, aux1_y=None, aux2_y=None, aux3_y=None,
                    x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
                    early_stopping_patience=30, record_loss=False):
        """
        Train method for the multi-task neural network.
        Records only main-task BCE loss for train/validation if record_loss=True.
        Uses early stopping on the main task validation loss.

        Args:
            x (Tensor): Training features.
            main_y (Tensor): Main binary labels.
            aux1_y, aux2_y (Tensor): Aux regression targets.
            aux3_y (Tensor): Aux binary target.
            x_val (Tensor, optional): Validation features.
            main_y_val (Tensor, optional): Validation main labels.
            aux1_val, aux2_val, aux3_val: Validation aux targets.
            early_stopping_patience (int): Patience on main val loss.
            record_loss (bool): If True, return (train_loss_history, val_loss_history).

        Returns:
            If record_loss=True: train_loss_history, val_loss_history lists.
        """
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            main_out, a1_out, a2_out, a3_out = self.forward(x)

            # Main task loss
            loss_main = self.main_loss_fn(main_out, main_y.view(-1))

            # Total loss = main + λ * (aux losses)
            if aux1_y is not None:
                loss_aux1 = self.aux_loss_fn(a1_out, aux1_y.view(-1))
                loss_aux2 = self.aux_loss_fn(a2_out, aux2_y.view(-1))
                loss_aux3 = self.aux_task3_loss_fn(a3_out, aux3_y.view(-1))
                total_loss = loss_main + self.lambda_aux * (loss_aux1 + loss_aux2 + loss_aux3)
            else:
                total_loss = loss_main

            total_loss.backward()
            self.optimizer.step()

            if record_loss:
                train_loss_history.append(loss_main.item())

            # Validation step (main task only)
            if x_val is not None and main_y_val is not None:
                self.eval()
                with torch.no_grad():
                    main_out_val, _, _, _ = self.forward(x_val)
                    val_loss_main = self.main_loss_fn(main_out_val, main_y_val.view(-1))
                if record_loss:
                    val_loss_history.append(val_loss_main.item())

                # Early stopping on main validation loss
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
        Evaluate the model on the main task only.
        Returns AUC, per-class precision/recall/F1, and main BCE loss.
        """
        self.eval()
        with torch.no_grad():
            main_out, _, _, _ = self.forward(x)
            preds = (main_out >= 0.5).float().cpu().numpy()
            y_np = y.cpu().numpy()

            p, r, f, _ = precision_recall_fscore_support(
                y_np, preds, labels=[0, 1], zero_division=0
            )
            auc = roc_auc_score(y_np, main_out.cpu().numpy())
            bce_loss = self.main_loss_fn(main_out, y.view(-1)).item()

        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
            "bce_loss": bce_loss
        }


class MultiTaskNN_PretrainFinetuneExtended(nn.Module):
    """
    Two-phase multi-task network:
      - Phase 1: Pre-train on auxiliary losses only.
      - Phase 2: Fine-tune on main task (with optional validation and early stopping).
      Records pretrain, finetune, and validation losses if requested.
    """

    def __init__(self, input_dim, hidden_dim=16, num_layers=1,
                 lr_pre=0.01, lr_fine=0.005, lambda_aux=0.3,
                 pre_epochs=100, fine_epochs=100):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension for feature layers.
            num_layers (int): Number of intermediate layers.
            lr_pre (float): Learning rate for pre-training.
            lr_fine (float): Learning rate for fine-tuning.
            lambda_aux (float): Weight for auxiliary losses.
            pre_epochs (int): Number of pre-training epochs.
            fine_epochs (int): Number of fine-tuning epochs.
        """
        super(MultiTaskNN_PretrainFinetuneExtended, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lambda_aux = lambda_aux
        self.pre_epochs = pre_epochs
        self.fine_epochs = fine_epochs

        # Build feature extractor layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature = nn.Sequential(*layers)

        # Shared layer after feature extractor
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # Heads
        self.main = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.aux1 = nn.Linear(hidden_dim, 1)
        self.aux2 = nn.Linear(hidden_dim, 1)
        self.aux3 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # Attention parameters (learnable α) for each intermediate layer output
        self.alpha1 = nn.Parameter(torch.ones(num_layers))
        self.alpha2 = nn.Parameter(torch.ones(num_layers))
        self.alpha3 = nn.Parameter(torch.ones(num_layers))

        # Loss functions
        self.main_loss_fn = nn.BCELoss()
        self.aux_mse = nn.MSELoss()
        self.aux_bce = nn.BCELoss()

        # Learning rates
        self.lr_pre = lr_pre
        self.lr_fine = lr_fine

    def forward(self, x):
        """
        Forward pass:
          - Collect outputs from each linear layer for skip connections.
          - Compute main head on last intermediate output.
          - Compute auxiliary heads on weighted sums of intermediate outputs.
        Returns:
            main_out, aux1_out, aux2_out, aux3_out
        """
        outs = []
        h = x
        # Pass through feature extractor and record each Linear output
        for layer in self.feature:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                outs.append(h)

        # Main head on last layer output
        shared_main = self.shared(outs[-1])
        main_out = self.main(shared_main).view(-1)

        # Weighted sum helper
        def weighted(alpha, seq):
            w = torch.softmax(alpha, dim=0)
            return sum(w[i] * seq[i] for i in range(len(seq)))

        s1 = self.shared(weighted(self.alpha1, outs))
        s2 = self.shared(weighted(self.alpha2, outs))
        s3 = self.shared(weighted(self.alpha3, outs))

        a1 = self.aux1(s1).view(-1)
        a2 = self.aux2(s2).view(-1)
        a3 = self.aux3(s3).view(-1)

        return main_out, a1, a2, a3

    def train_pre_fine(self,
                      x_pre, y_main_pre, aux1_pre, aux2_pre, aux3_pre,
                      x_fine, y_fine,
                      record_loss=False, early_stopping_patience=30,
                      x_fine_val=None, y_fine_val=None):
        """
        Two-phase training:
          Phase 1: Pre-train on auxiliary tasks using x_pre and aux targets.
          Phase 2: Fine-tune on main task using x_fine and main labels, with optional validation.
        
        Args:
            x_pre (Tensor): Features for pre-training.
            y_main_pre (Tensor): Main labels for pre-training (unused but kept for API consistency).
            aux1_pre, aux2_pre, aux3_pre: Auxiliary targets for pre-training.
            x_fine (Tensor): Features for fine-tuning on main task.
            y_fine (Tensor): Main labels for fine-tuning.
            record_loss (bool): If True, returns loss histories.
            early_stopping_patience (int): Patience for main-task early stopping.
            x_fine_val (Tensor): Validation features for main task.
            y_fine_val (Tensor): Validation labels for main task.
        
        Returns:
            If record_loss=True: (pre_losses, fine_losses, val_losses).
        """
        pre_losses = []
        fine_losses = []
        val_losses = []

        # --- Phase 1: Pre-training on auxiliary tasks ---
        opt_pre = optim.Adam(
            list(self.feature.parameters()) +
            list(self.aux1.parameters()) +
            list(self.aux2.parameters()) +
            list(self.aux3.parameters()) +
            [self.alpha1, self.alpha2, self.alpha3],
            lr=self.lr_pre
        )

        for _ in range(self.pre_epochs):
            self.train()
            opt_pre.zero_grad()
            _, o1, o2, o3 = self.forward(x_pre)
            l1 = self.aux_mse(o1, aux1_pre.view(-1))
            l2 = self.aux_mse(o2, aux2_pre.view(-1))
            l3 = self.aux_bce(o3, aux3_pre.view(-1))
            loss_side = self.lambda_aux * (l1 + l2 + l3)
            loss_side.backward()
            opt_pre.step()
            if record_loss:
                pre_losses.append(loss_side.item())

        # Freeze auxiliary components
        self.alpha1.requires_grad = False
        self.alpha2.requires_grad = False
        self.alpha3.requires_grad = False
        for p in self.aux1.parameters(): p.requires_grad = False
        for p in self.aux2.parameters(): p.requires_grad = False
        for p in self.aux3.parameters(): p.requires_grad = False

        # --- Phase 2: Fine-tuning on main task ---
        opt_fine = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr_fine)
        best_val = float('inf')
        patience = 0

        for epoch in range(self.fine_epochs):
            self.train()
            opt_fine.zero_grad()
            out_main, _, _, _ = self.forward(x_fine)
            loss_main = self.main_loss_fn(out_main, y_fine.view(-1))
            loss_main.backward()
            opt_fine.step()
            if record_loss:
                fine_losses.append(loss_main.item())

            # Validation step on main task
            if x_fine_val is not None and y_fine_val is not None:
                self.eval()
                with torch.no_grad():
                    v_main, _, _, _ = self.forward(x_fine_val)
                    v_loss = self.main_loss_fn(v_main, y_fine_val.view(-1))
                if record_loss:
                    val_losses.append(v_loss.item())
                if v_loss.item() < best_val:
                    best_val = v_loss.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break

        if record_loss:
            return pre_losses, fine_losses, val_losses

    def train_model(
        self,
        x, main_y,
        aux1_y=None, aux2_y=None, aux3_y=None,
        x_val=None, main_y_val=None,
        aux1_val=None, aux2_val=None, aux3_val=None,
        early_stopping_patience=30, record_loss=False
    ):
        """
        Wrapper that calls train_pre_fine for this two-phase model.
        """
        return self.train_pre_fine(
            x_pre=x,
            y_main_pre=main_y,
            aux1_pre=aux1_y,
            aux2_pre=aux2_y,
            aux3_pre=aux3_y,
            x_fine=x,
            y_fine=main_y,
            record_loss=record_loss,
            early_stopping_patience=early_stopping_patience,
            x_fine_val=x_val,
            y_fine_val=main_y_val
        )

    def evaluate(self, x, y):
        """
        Evaluate the model on the main task.
        Returns AUC, per-class precision/recall/F1, and BCE loss.
        """
        self.eval()
        with torch.no_grad():
            out_main, _, _, _ = self.forward(x)
            preds = (out_main >= 0.5).float().cpu().numpy()
            y_np = y.cpu().numpy()

            p, r, f, _ = precision_recall_fscore_support(
                y_np, preds, labels=[0, 1], zero_division=0
            )
            auc = roc_auc_score(y_np, out_main.cpu().numpy())
            loss = self.main_loss_fn(out_main, y.view(-1)).item()

        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
            "bce_loss": loss
        }


class MultiTaskNN_Decoupled(nn.Module):
    """
    Decoupled multi-task network:
      - Phase 1: Optimize auxiliary losses for φ & β.
      - Phase 2: Freeze φ & β, then optimize main head ψ with optional validation.
    """

    def __init__(self, input_dim, hidden_dim=16, num_layers=1,
                 lr=0.01, lambda_aux=0.3,
                 pre_epochs=100, main_epochs=100):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension for feature layers.
            num_layers (int): Number of intermediate layers.
            lr (float): Learning rate for both phases.
            lambda_aux (float): Weight for auxiliary losses.
            pre_epochs (int): Number of epochs for auxiliary optimization.
            main_epochs (int): Number of epochs for main-task optimization.
        """
        super(MultiTaskNN_Decoupled, self).__init__()
        self.lambda_aux = lambda_aux
        self.pre_epochs = pre_epochs
        self.main_epochs = main_epochs
        self.lr = lr

        # Build feature extractor
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature = nn.Sequential(*layers)

        # Shared layer
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # Heads
        self.main = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.aux1 = nn.Linear(hidden_dim, 1)
        self.aux2 = nn.Linear(hidden_dim, 1)
        self.aux3 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # Attention α parameters
        self.alpha1 = nn.Parameter(torch.ones(num_layers))
        self.alpha2 = nn.Parameter(torch.ones(num_layers))
        self.alpha3 = nn.Parameter(torch.ones(num_layers))

        # Loss functions
        self.main_loss_fn = nn.BCELoss()
        self.aux_mse = nn.MSELoss()
        self.aux_bce = nn.BCELoss()

    def forward(self, x):
        """
        Forward pass:
          - Collect intermediate outputs from feature extractor.
          - Compute main head on last intermediate output.
          - Compute auxiliary heads on weighted sums of intermediate outputs.
        Returns:
            main_out, aux1_out, aux2_out, aux3_out
        """
        outs = []
        h = x
        for layer in self.feature:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                outs.append(h)

        shared_main = self.shared(outs[-1])
        main_out = self.main(shared_main).view(-1)

        def weighted(alpha, seq):
            w = torch.softmax(alpha, dim=0)
            return sum(w[i] * seq[i] for i in range(len(seq)))

        s1 = self.shared(weighted(self.alpha1, outs))
        s2 = self.shared(weighted(self.alpha2, outs))
        s3 = self.shared(weighted(self.alpha3, outs))

        a1 = self.aux1(s1).view(-1)
        a2 = self.aux2(s2).view(-1)
        a3 = self.aux3(s3).view(-1)

        return main_out, a1, a2, a3

    def train_model(
        self,
        x, main_y, aux1_y, aux2_y, aux3_y,
        x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
        early_stopping_patience=30,
        record_loss=False
    ):
        """
        Two-phase training:
          Phase 1: Optimize auxiliary losses (φ & β) on all training data.
          Phase 2: Freeze φ & β, then optimize main head ψ on the same data or validation.

        Args:
            x (Tensor): Features for both phases.
            main_y (Tensor): Main labels.
            aux1_y, aux2_y, aux3_y (Tensor): Auxiliary targets.
            x_val, main_y_val (Tensor): Validation for main phase.
            early_stopping_patience (int): Patience for early stopping on main task.
            record_loss (bool): If True, print or return loss histories.
        Returns:
            If record_loss=True: prints main_losses and pre_losses.
        """
        pre_losses = []
        main_losses = []
        val_losses = []

        # --- Phase 1: Auxiliary Optimization ---
        opt_pre = optim.Adam(
            list(self.feature.parameters()) +
            list(self.aux1.parameters()) +
            list(self.aux2.parameters()) +
            list(self.aux3.parameters()) +
            [self.alpha1, self.alpha2, self.alpha3],
            lr=self.lr
        )
        for _ in range(self.pre_epochs):
            self.train()
            opt_pre.zero_grad()
            _, o1, o2, o3 = self.forward(x)
            loss_side = self.lambda_aux * (
                self.aux_mse(o1, aux1_y.view(-1)) +
                self.aux_mse(o2, aux2_y.view(-1)) +
                self.aux_bce(o3, aux3_y.view(-1))
            )
            loss_side.backward()
            opt_pre.step()
            if record_loss:
                pre_losses.append(loss_side.item())

        # Freeze φ & β (feature extractor, shared, and aux heads)
        for p in self.feature.parameters():
            p.requires_grad = False
        for p in self.shared.parameters():
            p.requires_grad = False
        for p in self.aux1.parameters():
            p.requires_grad = False
        for p in self.aux2.parameters():
            p.requires_grad = False
        for p in self.aux3.parameters():
            p.requires_grad = False
        self.alpha1.requires_grad = False
        self.alpha2.requires_grad = False
        self.alpha3.requires_grad = False

        # --- Phase 2: Main Task Optimization with Validation ---
        opt_main = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        best_val = float('inf')
        patience = 0

        for _ in range(self.main_epochs):
            self.train()
            opt_main.zero_grad()
            out_main, _, _, _ = self.forward(x)
            loss_main = self.main_loss_fn(out_main, main_y.view(-1))
            loss_main.backward()
            opt_main.step()
            if record_loss:
                main_losses.append(loss_main.item())

            if x_val is not None and main_y_val is not None:
                self.eval()
                with torch.no_grad():
                    v_main, _, _, _ = self.forward(x_val)
                    v_loss = self.main_loss_fn(v_main, main_y_val.view(-1))
                if record_loss:
                    val_losses.append(v_loss.item())
                if v_loss.item() < best_val:
                    best_val = v_loss.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break
            self.train()

        if record_loss:
            print("Main losses:", main_losses)
            print("Pre-training losses:", pre_losses)
            return main_losses, val_losses

    def evaluate(self, x, y):
        """
        Evaluate the model on the main task.
        Returns AUC, precision/recall/F1 for each class, and BCE loss.
        """
        self.eval()
        with torch.no_grad():
            out_main, _, _, _ = self.forward(x)
            preds = (out_main >= 0.5).float().cpu().numpy()
            y_np = y.cpu().numpy()

            p, r, f, _ = precision_recall_fscore_support(y_np, preds,
                                                         labels=[0, 1], zero_division=0)
            auc = roc_auc_score(y_np, out_main.cpu().numpy())
            loss = self.main_loss_fn(out_main, y.view(-1)).item()

        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
            "bce_loss": loss
        }

# =========================
# run_experiments_multitask
# =========================
def run_experiments_multitask(
    model_class,
    method_name: str,
    num_runs: int,
    param_grid: dict,
    train_x_pool: torch.Tensor,
    train_y_pool: torch.Tensor,
    train_aux1_pool: torch.Tensor,
    train_aux2_pool: torch.Tensor,
    train_aux3_pool: torch.Tensor,
    test_x_pool: torch.Tensor,
    test_y_pool: torch.Tensor,
    cv: int = 3
) -> (list, list):
    """
    Combined function that, for each iteration:
      1. Samples 250 positives + 250 negatives *without replacement* from train_pool → 500 train samples.
      2. Samples 125 test examples (may overlap across iterations) from test_pool.
      3. Performs 3-fold CV on those 500 train samples (including auxiliary targets) to find best hyperparameters.
      4. Retrains model on all 500 train samples using best hyperparameters and all auxiliary targets.
      5. Evaluates the retrained model on the 125-sample test set → collects metrics.
      6. Repeats num_runs times, ensuring that each 500-sample training set does not overlap across iterations.

    Returns:
        metrics_list (List[dict]): One dict of evaluation metrics (main task) per iteration.
        best_params_list (List[dict]): The best_params dict found in each iteration.
    """
    # Lists to store per-iteration results
    metrics_list = []
    best_params_list = []

    # Build DataFrame for sampling without overlap
    df_pool = pd.DataFrame(train_x_pool.numpy())
    df_pool['Diabetes_binary'] = train_y_pool.numpy()
    class1_indices = df_pool[df_pool['Diabetes_binary'] == 1].index.tolist()
    class0_indices = df_pool[df_pool['Diabetes_binary'] == 0].index.tolist()

    # Keep track of which indices have already been used in a previous iteration
    used_indices = set()
    test_pool_size = len(test_x_pool)

    # Signature of model class to know which hyperparams to pass
    sig = inspect.signature(model_class.__init__).parameters

    # Ensure image directory exists
    os.makedirs("img", exist_ok=True)

    for run_idx in range(1, num_runs + 1):
        # ----------------------------
        # 1) Balanced sampling: 250 positives and 250 negatives, no overlap
        # ----------------------------
        available_pos = [i for i in class1_indices if i not in used_indices]
        available_neg = [i for i in class0_indices if i not in used_indices]

        sampled_pos = random.sample(available_pos, 250)
        sampled_neg = random.sample(available_neg, 250)
        train_idx = sampled_pos + sampled_neg
        used_indices.update(train_idx)

        x_train = train_x_pool[train_idx]
        y_train = train_y_pool[train_idx]
        a1_train = train_aux1_pool[train_idx]
        a2_train = train_aux2_pool[train_idx]
        a3_train = train_aux3_pool[train_idx]

        # ----------------------------
        # 2) Sample random subset of test pool (125 examples, can overlap across runs)
        # ----------------------------
        test_idx = random.sample(range(test_pool_size), 125)
        x_test = test_x_pool[test_idx]
        y_test = test_y_pool[test_idx]

        # ----------------------------
        # 3) 3-fold CV for grid search (with auxiliary targets)
        # ----------------------------
        best_auc = -1.0
        best_params = None
        keys = list(param_grid.keys())

        X_np = x_train.numpy()
        y_np = y_train.numpy()
        # Convert auxiliary targets to numpy as well
        a1_np = a1_train.numpy()
        a2_np = a2_train.numpy()
        a3_np = a3_train.numpy()

        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for values in itertools.product(*[param_grid[k] for k in keys]):
            params = dict(zip(keys, values))
            fold_aucs = []

            # For each fold, split x_train into train/val
            for tr_idx, val_idx in kf.split(X_np):
                x_tr = torch.tensor(X_np[tr_idx], dtype=torch.float32)
                y_tr = torch.tensor(y_np[tr_idx], dtype=torch.float32)
                x_val = torch.tensor(X_np[val_idx], dtype=torch.float32)
                y_val = torch.tensor(y_np[val_idx], dtype=torch.float32)

                # Prepare auxiliary splits
                a1_tr = torch.tensor(a1_np[tr_idx], dtype=torch.float32)
                a2_tr = torch.tensor(a2_np[tr_idx], dtype=torch.float32)
                a3_tr = torch.tensor(a3_np[tr_idx], dtype=torch.float32)
                a1_val = torch.tensor(a1_np[val_idx], dtype=torch.float32)
                a2_val = torch.tensor(a2_np[val_idx], dtype=torch.float32)
                a3_val = torch.tensor(a3_np[val_idx], dtype=torch.float32)

                # Instantiate model with given hyperparameters
                init_kwargs = {}
                if 'input_dim' in sig:
                    init_kwargs['input_dim'] = X_np.shape[1]
                for k, v in params.items():
                    if k in sig:
                        init_kwargs[k] = v
                model = model_class(**init_kwargs)

                # Train on the fold (with auxiliary targets)
                model.train_model(
                    x_tr, y_tr,
                    aux1_y=a1_tr, aux2_y=a2_tr, aux3_y=a3_tr,
                    x_val=x_val, main_y_val=y_val,
                    aux1_val=a1_val, aux2_val=a2_val, aux3_val=a3_val,
                    early_stopping_patience=10,
                    record_loss=False
                )

                # Compute validation AUC (main task only)
                with torch.no_grad():
                    main_out = model.forward(x_val)[0]
                    auc_score = roc_auc_score(y_val.cpu().numpy(),
                                              main_out.cpu().numpy())
                    fold_aucs.append(auc_score)

            avg_auc = float(np.mean(fold_aucs))
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_params = params.copy()

        print(f"[{method_name}] Iteration {run_idx} → Best CV params: {best_params}, Avg CV AUC: {best_auc:.4f}")
        best_params_list.append(best_params)

        # ----------------------------
        # 4) Retrain on all 500 samples with best_params
        # ----------------------------
        init_kwargs = {}
        if 'input_dim' in sig:
            init_kwargs['input_dim'] = x_train.shape[1]
        for k, v in best_params.items():
            if k in sig:
                init_kwargs[k] = v
        best_model = model_class(**init_kwargs)

        best_model.train_model(
            x_train, y_train,
            aux1_y=a1_train, aux2_y=a2_train, aux3_y=a3_train,
            early_stopping_patience=10,
            record_loss=False
        )

        # ----------------------------
        # 5) Evaluate on the 125-sample test set
        # ----------------------------
        metrics = best_model.evaluate(x_test, y_test)
        print(f"[{method_name}] Iteration {run_idx} → Test metrics: {metrics}")
        metrics_list.append(metrics)

    return metrics_list, best_params_list


# =========================
#   AGGREGATION & UTILITIES
# =========================

def aggregate_metrics(metrics_list):
    """
    Aggregate a list of metric dictionaries by computing mean ± std for each metric key.
    """
    agg = {}
    for key in metrics_list[0].keys():
        vals = [m[key] for m in metrics_list]
        mean_val = float(torch.tensor(vals).mean().item())
        std_val = float(torch.tensor(vals).std().item())
        agg[key] = f"{mean_val:.4f} ± {std_val:.4f}"
    return agg


def write_results_csv(filename, method, metrics, params):
    """
    Append a row of results (metrics + params) to a CSV file.
    """
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            method,
            metrics.get('auc', ''),
            metrics.get('p0', ''), metrics.get('r0', ''), metrics.get('f0', ''),
            metrics.get('p1', ''), metrics.get('r1', ''), metrics.get('f1', ''),
            metrics.get('bce_loss', ''),
            params
        ])


def print_aggregated_results(method, metrics, params):
    """
    Print aggregated results in a readable format.
    """
    print("\n=== Final Aggregated Results ===")
    print(f"Method: {method}")
    print(f"AUC: {metrics.get('auc','')}")
    print(f"Class0 → Precision: {metrics.get('p0','')}, Recall: {metrics.get('r0','')}, F1: {metrics.get('f0','')}")
    print(f"Class1 → Precision: {metrics.get('p1','')}, Recall: {metrics.get('r1','')}, F1: {metrics.get('f1','')}")
    print(f"Main BCE Loss: {metrics.get('bce_loss','')}")
    print(f"Best Parameters: {params}")
    print("---------------------------------------------------")


# =========================
#          MAIN
# =========================
def main():
    """
    Main entry point to run multi-task experiments.
    Loads data, runs per-iteration grid search + retraining, and saves results.
    """
    # 1. Ensure reproducibility
    set_seed(42)

    OUTPUT_CSV = "multitask_results.csv"
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Method", "AUC",
                "Prec0", "Rec0", "F10",
                "Prec1", "Rec1", "F11",
                "BCE_Loss", "Params"
            ])

    # 2. Preprocess data
    dp = MultiTaskDatasetPreprocessor(
        dataset_id=891,
        side_info_path='prompting/augmented_data_70B.csv'
    )
    (train_x, train_y, ta1, ta2, ta3,
     test_x, test_y) = dp.preprocess()

    # 3. Define experiments
    experiments = [
        # {
        #     "name": "MT-LR",
        #     "model": MultiTaskLogisticRegression,
        #     "param_grid": {"lr": [0.001, 0.01]}
        # }
        # ,
        # {
        #     "name": "MT-MLP",
        #     "model": MultiTaskNN,
        #     "param_grid": {
        #         "lr": [0.01],
        #         "hidden_dim": [128],
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
        #         "lambda_aux": [0.01, 0.1, 0.3]
        #     }
        # }
        # ,
        # {
        #     "name": "PretrainFT",
        #     "model": MultiTaskNN_PretrainFinetuneExtended,
        #     "param_grid": {
        #         "lr_pre": [0.01, 0.1],
        #         "lr_fine": [0.01, 0.1],
        #         "num_layers": [1,2,3,4],
        #         "hidden_dim":[64, 128, 256],
        #         "lambda_aux": [0.01, 0.3, 0.1],
        #         "pre_epochs": [100, 300],
        #         "fine_epochs": [100, 300]
        #     }
        # }
        # ,
        {
            "name": "Decoupled",
            "model": MultiTaskNN_Decoupled,
            "param_grid": {
                "lr": [0.01, 0.1],
                "hidden_dim": [64, 128, 256],
                "num_layers": [1, 2, 3, 4],
                "lambda_aux": [0.01, 0.1, 0.3]
            }
        }
    ]

    # 4. Loop over each experiment configuration
    for cfg in experiments:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        name = cfg["name"]
        Model = cfg["model"]
        param_grid = cfg["param_grid"]

        # 5. Run per-iteration grid search + retraining
        results, best_params_list = run_experiments_multitask(
            model_class=Model,
            method_name=name,
            num_runs=10,
            param_grid=param_grid,
            train_x_pool=train_x,
            train_y_pool=train_y,
            train_aux1_pool=ta1,
            train_aux2_pool=ta2,
            train_aux3_pool=ta3,
            test_x_pool=test_x,
            test_y_pool=test_y,
            cv=3
        )

        # 6. Aggregate metrics and print/save
        agg = aggregate_metrics(results)
        # Store only the 1st iteration's best_params for summary CSV (or average / custom aggregation)
        write_results_csv(OUTPUT_CSV, name, agg, str(best_params_list))
        print_aggregated_results(name, agg, str(best_params_list))


if __name__ == "__main__":
    main()