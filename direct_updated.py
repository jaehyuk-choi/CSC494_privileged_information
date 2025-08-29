# # direct_experiments.py

# import os
# import csv
# import random
# import itertools
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# from ucimlrepo import fetch_ucirepo


# # =========================
# #    GLOBAL SEED FUNCTION
# # =========================

# def set_seed(seed: int):
#     """
#     Set random seed for reproducibility across random, NumPy, and PyTorch.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# # =========================
# #   DATA PREPROCESSOR
# # =========================

# class DirectDataPreprocessor:
#     """
#     Preprocessor for Direct-pattern models.
#     Uses LLM-augmented side information (privileged) during training,
#     but excludes it from the test set.
#     """

#     def __init__(
#         self,
#         dataset_id: int = 891,
#         side_info_path: str = "prompting/augmented_data.csv",
#     ):
#         """
#         Args:
#             dataset_id (int): UCI repository dataset ID (default: 891).
#             side_info_path (str): Path to CSV containing LLM-augmented side information.
#         """
#         # Fetch original UCI dataset
#         self.original_data = fetch_ucirepo(id=dataset_id)
#         self.X_original = self.original_data.data.features  # pandas DataFrame of features
#         self.y_original = self.original_data.data.targets   # pandas Series of main labels

#         # StandardScaler for continuous features (BMI and side info)
#         self.scaler = StandardScaler()

#         # Store path to LLM-generated side information CSV
#         self.side_path = side_info_path

#         # Columns used as input features
#         self.training_cols = [
#             "HighBP",
#             "HighChol",
#             "CholCheck",
#             "BMI",
#             "Smoker",
#             "Stroke",
#             "HeartDiseaseorAttack",
#             "PhysActivity",
#             "Fruits",
#             "Veggies",
#             "HvyAlcoholConsump",
#             "AnyHealthcare",
#             "NoDocbcCost",
#             "GenHlth",
#             "MentHlth",
#             "PhysHlth",
#             "DiffWalk",
#             "Sex",
#             "Age",
#             "Education",
#             "Income",
#         ]
#         # Column names for main target and privileged side information
#         self.target = "Diabetes_binary"
#         self.side_info = "predict_hba1c"  # LLM-generated privileged information

#     def preprocess(self):
#         """
#         Execute preprocessing and return tensors for:
#           - grid set (balanced) for hyperparameter tuning
#           - training pool (with side information)
#           - test set (excluded from side information)

#         Returns:
#             grid_x, grid_y, grid_z, train_x, train_y, train_z, test_x, test_y
#         """
#         continuous_cols = ["BMI", self.side_info]

#         # 1) Load augmented CSV (contains side information)
#         augmented_df = pd.read_csv(self.side_path)
#         augmented_df = augmented_df[self.training_cols + [self.target, self.side_info]]

#         # 2) Drop rows with missing main target or side information
#         augmented_df = augmented_df.dropna(subset=[self.target, self.side_info])

#         # 3) Create a balanced 'grid' set for hyperparameter tuning
#         pos_idx = augmented_df[augmented_df[self.target] == 1].index.tolist()
#         neg_idx = augmented_df[augmented_df[self.target] == 0].index.tolist()
#         n_pos = min(len(pos_idx), 375)
#         n_neg = min(len(neg_idx), 375)
#         grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
#         grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
#         grid_df = pd.concat([grid_pos, grid_neg])

#         # The remaining data after removing grid_df is used as training pool
#         train_pool = augmented_df.drop(index=grid_df.index)

#         # 4) Construct a separate test set using original UCI data (no side information)
#         original_df = self.X_original.copy()
#         original_df[self.target] = self.y_original
#         # Drop any rows that appear in augmented_df to avoid overlap
#         test_pool = original_df.drop(index=augmented_df.index, errors="ignore")
#         # Keep only feature columns and main target
#         test_pool = test_pool[self.training_cols + [self.target]]

#         # 5) Fit scaler on BMI and side_info from training pool, then transform grid and train_pool
#         self.scaler.fit(train_pool[continuous_cols])
#         grid_df_scaled = grid_df.copy()
#         grid_df_scaled[continuous_cols] = self.scaler.transform(grid_df[continuous_cols])
#         train_pool_scaled = train_pool.copy()
#         train_pool_scaled[continuous_cols] = self.scaler.transform(train_pool[continuous_cols])

#         # 6) For test set, only normalize BMI using scaler parameters (side_info not available)
#         bmi_mean = self.scaler.mean_[0]
#         bmi_scale = self.scaler.scale_[0]
#         test_pool = test_pool.copy()
#         test_pool["BMI"] = (test_pool["BMI"] - bmi_mean) / bmi_scale

#         # 7) Convert DataFrames to PyTorch tensors
#         def to_tensor(df: pd.DataFrame, multi: bool = False):
#             """
#             Convert features and targets (and side info if multi=True) to torch.Tensor.
#             """
#             x = torch.tensor(df[self.training_cols].values, dtype=torch.float32)
#             y = torch.tensor(df[self.target].values, dtype=torch.float32).view(-1, 1)
#             if multi:
#                 z = torch.tensor(df[self.side_info].values, dtype=torch.float32).view(-1, 1)
#                 return x, y, z
#             else:
#                 return x, y

#         # Tensors for grid set
#         grid_x, grid_y, grid_z = to_tensor(grid_df_scaled, multi=True)
#         # Tensors for training pool
#         train_x, train_y, train_z = to_tensor(train_pool_scaled, multi=True)
#         # Tensors for test set (no side_info)
#         test_x, test_y = to_tensor(test_pool, multi=False)

#         return grid_x, grid_y, grid_z, train_x, train_y, train_z, test_x, test_y


# # =========================
# #      MODEL CLASSES
# # =========================

# class DirectDecoupledResidualNoDecomp(nn.Module):
#     """
#     Decoupled Residual Model WITHOUT explicit expansion, 
#     with early stopping in fine-tune phase.
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int = 32,
#         num_layers: int = 3,
#         num_decoder_layers: int = 1,
#         residual_type: str = "concat",  # either "concat" or "sum"
#         lr: float = 0.001,
#         epochs: int = 100,
#     ):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             hidden_dim (int): Hidden dimension size.
#             num_layers (int): Number of encoder layers.
#             num_decoder_layers (int): Number of decoder layers.
#             residual_type (str): "concat" or "sum" for combining S and residual.
#             lr (float): Learning rate for both phases.
#             epochs (int): Number of epochs per phase.
#         """
#         super().__init__()
#         self.epochs = epochs
#         self.lr = lr
#         self.residual_type = residual_type

#         # Encoder: list of (Linear -> ReLU) modules
#         self.encoder_layers = nn.ModuleList()
#         dim = input_dim
#         for _ in range(num_layers):
#             self.encoder_layers.append(nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU()))
#             dim = hidden_dim

#         # Learnable weights for residual combination
#         self.alpha = nn.Parameter(torch.ones(num_layers))

#         # Linear projection of final encoder output to scalar S
#         self.to_S = nn.Linear(hidden_dim, 1)

#         # Decoder network: input dimension depends on residual_type
#         dec_in = hidden_dim + 1 if residual_type == "concat" else hidden_dim
#         decoder_layers = []
#         for _ in range(num_decoder_layers - 1):
#             decoder_layers.append(nn.Linear(dec_in, hidden_dim))
#             decoder_layers.append(nn.ReLU())
#             dec_in = hidden_dim
#         decoder_layers.append(nn.Linear(dec_in, 1))
#         decoder_layers.append(nn.Sigmoid())
#         self.decoder = nn.Sequential(*decoder_layers)

#         # Loss functions
#         self.loss_z = nn.MSELoss()  # for pretrain (MSE between S and privileged z)
#         self.loss_y = nn.BCELoss()  # for fine-tune (BCE between y_hat and y)

#     def forward(self, x: torch.Tensor):
#         """
#         Forward pass:
#           - Compute encoder outputs for each layer.
#           - Compute weighted sum of all encoder outputs as 'residual'.
#           - Project last encoder output to scalar S via to_S.
#           - Combine S and residual (sum or concat).
#           - Pass through decoder to get y_hat.
#         Returns:
#             S (Tensor [batch,1]), y_hat (Tensor [batch,1])
#         """
#         outs = []
#         h = x
#         for layer in self.encoder_layers:
#             h = layer(h)
#             outs.append(h)

#         # Softmax over alpha to weight each encoder output
#         w = torch.softmax(self.alpha, dim=0)
#         residual = sum(wi * oi for wi, oi in zip(w, outs))  # [batch, hidden_dim]

#         # Compute scalar S from the last encoder output
#         S = self.to_S(outs[-1])  # [batch, 1]

#         # Combine S and residual
#         if self.residual_type == "sum":
#             combined = S + residual  # broadcasting: [batch, hidden_dim]
#         else:  # "concat"
#             combined = torch.cat([S, residual], dim=1)  # [batch, hidden_dim+1]

#         # Decode to y_hat
#         y_hat = self.decoder(combined)  # [batch, 1]
#         return S, y_hat

#     def pretrain_phase(self, x: torch.Tensor, z: torch.Tensor, epochs: int = None, lr: float = None):
#         """
#         Phase 1: Pre-train encoder layers, to_S, and alpha by minimizing MSE(S, z).
#         """
#         e = epochs or self.epochs
#         lr0 = lr or self.lr
#         optimizer = optim.Adam(
#             list(self.encoder_layers.parameters()) + list(self.to_S.parameters()) + [self.alpha],
#             lr=lr0,
#         )
#         self.train()
#         for _ in range(e):
#             optimizer.zero_grad()
#             S_pred, _ = self.forward(x)
#             loss = self.loss_z(S_pred, z)
#             loss.backward()
#             optimizer.step()

#     def train_model(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         z: torch.Tensor,
#         x_val: torch.Tensor = None,
#         y_val: torch.Tensor = None,
#         z_val: torch.Tensor = None,
#         early_stopping: int = 10,
#         record_loss: bool = False,
#     ):
#         """
#         Full training: 
#           - Phase 1: Pre-train encoder + S projection.
#           - Phase 2: Freeze encoder components, train decoder with early stopping on BCE.

#         Args:
#             x (Tensor): Training features [N, D].
#             y (Tensor): Main labels [N, 1].
#             z (Tensor): Privileged side info [N, 1].
#             x_val, y_val, z_val (Tensor, optional): Validation sets.
#             early_stopping (int): Patience for early stopping on validation BCE loss.
#             record_loss (bool): If True, returns (train_history, val_history).
#         Returns:
#             (train_hist, val_hist) if record_loss else None
#         """
#         # Phase 1: pre-train
#         self.pretrain_phase(x, z)

#         # Freeze encoder components and S projection
#         for p in self.encoder_layers.parameters():
#             p.requires_grad = False
#         for p in self.to_S.parameters():
#             p.requires_grad = False
#         self.alpha.requires_grad = False

#         # Phase 2: fine-tune decoder only
#         optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
#         train_hist, val_hist = [], []
#         best_val = float("inf")
#         patience = 0

#         for epoch in range(self.epochs):
#             # Training step on decoder
#             self.train()
#             optimizer.zero_grad()
#             _, y_pred = self.forward(x)  # [N,1]
#             loss_train = self.loss_y(y_pred, y)
#             loss_train.backward()
#             optimizer.step()

#             if record_loss:
#                 train_hist.append(loss_train.item())

#             # Validation step for early stopping
#             if x_val is not None and y_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     _, yv_pred = self.forward(x_val)
#                     loss_val = self.loss_y(yv_pred, y_val)
#                 if record_loss:
#                     val_hist.append(loss_val.item())
#                 if loss_val.item() < best_val:
#                     best_val = loss_val.item()
#                     patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping:
#                         break

#         return (train_hist, val_hist) if record_loss else None

#     def evaluate(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         Evaluate on test set: compute AUC, BCE loss, and per-class precision/recall/F1.

#         Returns:
#             Dict with keys: "auc", "bce_loss", "p0", "r0", "f0", "p1", "r1", "f1"
#         """
#         self.eval()
#         with torch.no_grad():
#             _, y_pred = self.forward(x)  # [N,1]
#             y_pred_flat = y_pred.view(-1)
#             y_flat = y.view(-1)
#             preds = (y_pred_flat >= 0.5).float()

#             # Compute AUC (if possible)
#             try:
#                 auc_score = roc_auc_score(y_flat.cpu().numpy(), y_pred_flat.cpu().numpy())
#             except ValueError:
#                 auc_score = 0.0

#             bce_loss = self.loss_y(y_pred_flat, y_flat).item()
#             p, r, f, _ = precision_recall_fscore_support(
#                 y_flat.cpu().numpy(), preds.cpu().numpy(), zero_division=0
#             )

#         return {
#             "auc": auc_score,
#             "bce_loss": bce_loss,
#             "p0": p[0],
#             "r0": r[0],
#             "f0": f[0],
#             "p1": p[1],
#             "r1": r[1],
#             "f1": f[1],
#         }


# class DirectDecoupledResidualModel(nn.Module):
#     """
#     Decoupled Residual Model WITH explicit expansion,
#     with early stopping in fine-tune phase.
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int = 32,
#         num_layers: int = 3,
#         num_decoder_layers: int = 1,
#         residual_type: str = "concat",  # "concat" or "sum"
#         lr: float = 0.001,
#         epochs: int = 100,
#     ):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             hidden_dim (int): Hidden dimension size.
#             num_layers (int): Number of encoder layers.
#             num_decoder_layers (int): Number of decoder layers.
#             residual_type (str): "concat" or "sum".
#             lr (float): Learning rate for both phases.
#             epochs (int): Number of epochs per phase.
#         """
#         super().__init__()
#         self.epochs = epochs
#         self.lr = lr
#         self.residual_type = residual_type

#         # Encoder: list of (Linear -> ReLU) modules
#         self.encoder_layers = nn.ModuleList()
#         dim = input_dim
#         for _ in range(num_layers):
#             self.encoder_layers.append(nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU()))
#             dim = hidden_dim

#         # Learnable weights for residual combination
#         self.alpha = nn.Parameter(torch.ones(num_layers))

#         # Projection to scalar S and expansion back to hidden_dim
#         self.to_S = nn.Linear(hidden_dim, 1)
#         self.expand_S = nn.Linear(1, hidden_dim)

#         # Decoder network: input dimension depends on residual_type
#         dec_in = hidden_dim * 2 if residual_type == "concat" else hidden_dim
#         decoder_layers = []
#         for _ in range(num_decoder_layers - 1):
#             decoder_layers.append(nn.Linear(dec_in, hidden_dim))
#             decoder_layers.append(nn.ReLU())
#             dec_in = hidden_dim
#         decoder_layers.append(nn.Linear(dec_in, 1))
#         decoder_layers.append(nn.Sigmoid())
#         self.decoder = nn.Sequential(*decoder_layers)

#         # Loss functions
#         self.loss_z = nn.MSELoss()  # for pretrain
#         self.loss_y = nn.BCELoss()  # for fine-tune

#     def forward(self, x: torch.Tensor):
#         """
#         Forward pass:
#           - Compute encoder outputs for each layer.
#           - Compute weighted sum of encoder outputs as 'residual'.
#           - Project last encoder output to S, then expand to hidden_dim.
#           - Combine expanded S with residual.
#           - Decode to y_hat.
#         Returns:
#             S (Tensor [batch,1]), y_hat (Tensor [batch,1])
#         """
#         outs = []
#         h = x
#         for layer in self.encoder_layers:
#             h = layer(h)
#             outs.append(h)

#         # Weighted residual
#         w = torch.softmax(self.alpha, dim=0)
#         residual = sum(wi * oi for wi, oi in zip(w, outs))  # [batch, hidden_dim]

#         # Project to scalar S and expand back
#         S = self.to_S(outs[-1])            # [batch, 1]
#         exp_S = self.expand_S(S)           # [batch, hidden_dim]

#         # Combine expanded S and residual
#         if self.residual_type == "sum":
#             combined = exp_S + residual  # [batch, hidden_dim]
#         else:  # "concat"
#             combined = torch.cat([exp_S, residual], dim=1)  # [batch, hidden_dim*2]

#         # Decode to y_hat
#         y_hat = self.decoder(combined)  # [batch, 1]
#         return S, y_hat

#     def pretrain_phase(self, x: torch.Tensor, z: torch.Tensor, epochs: int = None, lr: float = None):
#         """
#         Phase 1: Train encoder, to_S, expand_S, and alpha to minimize MSE(S, z).
#         """
#         e = epochs or self.epochs
#         lr0 = lr or self.lr
#         optimizer = optim.Adam(
#             list(self.encoder_layers.parameters())
#             + list(self.to_S.parameters())
#             + list(self.expand_S.parameters())
#             + [self.alpha],
#             lr=lr0,
#         )
#         self.train()
#         for _ in range(e):
#             optimizer.zero_grad()
#             S_pred, _ = self.forward(x)
#             loss = self.loss_z(S_pred, z)
#             loss.backward()
#             optimizer.step()

#     def train_model(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         z: torch.Tensor,
#         x_val: torch.Tensor = None,
#         y_val: torch.Tensor = None,
#         z_val: torch.Tensor = None,
#         early_stopping: int = 10,
#         record_loss: bool = False,
#     ):
#         """
#         Full training:
#           - Phase 1: Pretrain encoder + to_S + expand_S + alpha.
#           - Phase 2: Freeze encoder components, train decoder with early stopping on BCE.

#         Args:
#             x (Tensor): Training features [N, D].
#             y (Tensor): Main labels [N, 1].
#             z (Tensor): Privileged side info [N, 1].
#             x_val, y_val, z_val (Tensor, optional): Validation sets.
#             early_stopping (int): Patience for early stopping on validation BCE loss.
#             record_loss (bool): If True, returns (train_history, val_history).
#         Returns:
#             (train_hist, val_hist) if record_loss else None
#         """
#         # Phase 1: pretrain
#         self.pretrain_phase(x, z)

#         # Freeze encoder, to_S, expand_S, and alpha
#         for p in self.encoder_layers.parameters():
#             p.requires_grad = False
#         for p in self.to_S.parameters():
#             p.requires_grad = False
#         for p in self.expand_S.parameters():
#             p.requires_grad = False
#         self.alpha.requires_grad = False

#         # Phase 2: train decoder
#         optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
#         train_hist, val_hist = [], []
#         best_val = float("inf")
#         patience = 0

#         for epoch in range(self.epochs):
#             # Training step
#             self.train()
#             optimizer.zero_grad()
#             _, y_pred = self.forward(x)
#             loss_train = self.loss_y(y_pred, y)
#             loss_train.backward()
#             optimizer.step()

#             if record_loss:
#                 train_hist.append(loss_train.item())

#             # Validation step
#             if x_val is not None and y_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     _, yv_pred = self.forward(x_val)
#                     loss_val = self.loss_y(yv_pred, y_val)
#                 if record_loss:
#                     val_hist.append(loss_val.item())
#                 if loss_val.item() < best_val:
#                     best_val = loss_val.item()
#                     patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping:
#                         break

#         return (train_hist, val_hist) if record_loss else None

#     def evaluate(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         Evaluate on test set: compute AUC, BCE loss, and per-class precision/recall/F1.

#         Returns:
#             Dict with keys: "auc", "bce_loss", "p0", "r0", "f0", "p1", "r1", "f1"
#         """
#         self.eval()
#         with torch.no_grad():
#             _, y_pred = self.forward(x)
#             y_pred_flat = y_pred.view(-1)
#             y_flat = y.view(-1)
#             preds = (y_pred_flat >= 0.5).float()

#             # Compute AUC
#             try:
#                 auc_score = roc_auc_score(y_flat.cpu().numpy(), y_pred_flat.cpu().numpy())
#             except ValueError:
#                 auc_score = 0.0

#             bce_loss = self.loss_y(y_pred_flat, y_flat).item()
#             p, r, f, _ = precision_recall_fscore_support(
#                 y_flat.cpu().numpy(), preds.cpu().numpy(), zero_division=0
#             )

#         return {
#             "auc": auc_score,
#             "bce_loss": bce_loss,
#             "p0": p[0],
#             "r0": r[0],
#             "f0": f[0],
#             "p1": p[1],
#             "r1": r[1],
#             "f1": f[1],
#         }


# class DirectPatternNoDecomp(nn.Module):
#     """
#     Direct Pattern Model WITHOUT explicit expansion layer.

#     - Encoder extracts features.
#     - to_S compresses to scalar S.
#     - Combine S with weighted sum of encoder outputs (residual).
#     - Decoder predicts y_hat.
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int = 32,
#         num_layers: int = 3,
#         num_decoder_layers: int = 1,
#         residual_type: str = "concat",  # "concat" or "sum"
#         lr: float = 0.001,
#         epochs: int = 100,
#     ):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             hidden_dim (int): Hidden dimension size.
#             num_layers (int): Number of encoder layers.
#             num_decoder_layers (int): Number of decoder layers.
#             residual_type (str): "concat" or "sum".
#             lr (float): Learning rate for joint training.
#             epochs (int): Number of epochs for joint training.
#         """
#         super().__init__()
#         self.epochs = epochs
#         self.residual_type = residual_type

#         # Encoder: list of (Linear -> ReLU) modules
#         self.feature_layers = nn.ModuleList()
#         for i in range(num_layers):
#             in_dim = input_dim if i == 0 else hidden_dim
#             self.feature_layers.append(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU()))

#         # Learnable weights for residual combination
#         self.alpha = nn.Parameter(torch.ones(num_layers))

#         # Projection to scalar S
#         self.to_S = nn.Linear(hidden_dim, 1)

#         # Decoder: input dim depends on residual_type
#         decoder_input = hidden_dim + 1 if residual_type == "concat" else hidden_dim
#         decoder_layers = []
#         for _ in range(num_decoder_layers - 1):
#             decoder_layers.append(nn.Linear(decoder_input, hidden_dim))
#             decoder_layers.append(nn.ReLU())
#             decoder_input = hidden_dim
#         decoder_layers.append(nn.Linear(decoder_input, 1))
#         decoder_layers.append(nn.Sigmoid())
#         self.decoder = nn.Sequential(*decoder_layers)

#         # Loss functions
#         self.loss_z = nn.MSELoss()  # MSE(S_pred, z)
#         self.loss_y = nn.BCELoss()  # BCE(y_pred, y)
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)

#     def forward(self, x: torch.Tensor):
#         """
#         Forward pass:
#           - Compute encoder outputs for each layer.
#           - Compute weighted sum of encoder outputs as 'residual'.
#           - Compute scalar S from last encoder output.
#           - Combine S and residual (sum or concat).
#           - Decode to y_hat.
#         Returns:
#             S (Tensor [batch,1]), y_hat (Tensor [batch,1])
#         """
#         outputs = []
#         h = x
#         for layer in self.feature_layers:
#             h = layer(h)
#             outputs.append(h)

#         weights = torch.softmax(self.alpha, dim=0)
#         residual = sum(w * o for w, o in zip(weights, outputs))  # [batch, hidden_dim]

#         S = self.to_S(outputs[-1])  # [batch, 1]

#         if self.residual_type == "sum":
#             combined = S + residual  # [batch, hidden_dim]
#         else:  # "concat"
#             combined = torch.cat([S, residual], dim=1)  # [batch, hidden_dim+1]

#         y_pred = self.decoder(combined)  # [batch, 1]
#         return S, y_pred

#     def train_model(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         z: torch.Tensor,
#         x_val: torch.Tensor = None,
#         y_val: torch.Tensor = None,
#         z_val: torch.Tensor = None,
#         early_stopping: int = 10,
#         record_loss: bool = False,
#     ):
#         """
#         Joint training with combined loss: MSE(S_pred, z) + BCE(y_pred, y).
#         Supports optional validation and early stopping on BCE.

#         Args:
#             x (Tensor): Training features [N, D].
#             y (Tensor): Main labels [N, 1].
#             z (Tensor): Privileged side info [N, 1].
#             x_val, y_val, z_val (Tensor, optional): Validation sets.
#             early_stopping (int): Patience for early stopping on validation BCE loss.
#             record_loss (bool): If True, returns (train_history, val_history).
#         Returns:
#             (train_hist, val_hist) if record_loss else None
#         """
#         train_hist, val_hist = [], []
#         best_val = float("inf")
#         patience = 0

#         for epoch in range(self.epochs):
#             self.train()
#             self.optimizer.zero_grad()
#             S_pred, y_pred = self.forward(x)
#             loss = self.loss_z(S_pred, z) + self.loss_y(y_pred, y)
#             loss.backward()
#             self.optimizer.step()

#             if record_loss:
#                 train_hist.append(self.loss_y(y_pred, y).item())

#             # Validation step
#             if x_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     _, yv_pred = self.forward(x_val)
#                     val_loss = self.loss_y(yv_pred, y_val).item()
#                 if record_loss:
#                     val_hist.append(val_loss)
#                 if val_loss < best_val:
#                     best_val = val_loss
#                     patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping:
#                         break

#         return (train_hist, val_hist) if record_loss else None

#     def evaluate(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         Evaluate on test set: compute accuracy, AUC, BCE loss, 
#         and per-class precision/recall/F1.

#         Returns:
#             Dict with keys: "accuracy", "auc", "bce_loss", 
#             "p0", "r0", "f0", "p1", "r1", "f1"
#         """
#         self.eval()
#         with torch.no_grad():
#             _, y_pred = self.forward(x)  # [N,1]
#             y_pred_flat = y_pred.view(-1)
#             y_flat = y.view(-1)
#             y_hat = (y_pred_flat >= 0.5).float()

#             # Accuracy
#             accuracy = (y_hat == y_flat).float().mean().item()

#             # AUC
#             try:
#                 auc_score = roc_auc_score(y_flat.cpu().numpy(), y_pred_flat.cpu().numpy())
#             except ValueError:
#                 auc_score = 0.0

#             bce_loss = self.loss_y(y_pred_flat, y_flat).item()
#             p, r, f, _ = precision_recall_fscore_support(
#                 y_flat.cpu().numpy(), y_hat.cpu().numpy(), zero_division=0
#             )

#         return {
#             "accuracy": accuracy,
#             "auc": auc_score,
#             "bce_loss": bce_loss,
#             "p0": p[0],
#             "r0": r[0],
#             "f0": f[0],
#             "p1": p[1],
#             "r1": r[1],
#             "f1": f[1],
#         }


# class DirectPatternResidual(nn.Module):
#     """
#     Direct Pattern Model WITH explicit expansion layer.

#     - Encoder compresses X to latent h.
#     - to_S compresses h to scalar S.
#     - expand_S expands S back to hidden dimension.
#     - Combine expanded S + last encoder output (sum or concat).
#     - Decoder predicts y_hat.
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int = 32,
#         num_layers: int = 3,
#         num_decoder_layers: int = 1,
#         residual_type: str = "concat",  # "concat" or "sum"
#         lr: float = 0.001,
#         epochs: int = 300,
#     ):
#         """
#         Args:
#             input_dim (int): Number of input features.
#             hidden_dim (int): Hidden dimension size.
#             num_layers (int): Number of encoder layers.
#             num_decoder_layers (int): Number of decoder layers.
#             residual_type (str): "concat" or "sum".
#             lr (float): Learning rate for joint training.
#             epochs (int): Number of epochs for joint training.
#         """
#         super().__init__()
#         self.epochs = epochs
#         self.residual_type = residual_type

#         # Build encoder layers
#         self.feature_layers = nn.ModuleList()
#         for i in range(num_layers):
#             in_dim = input_dim if i == 0 else hidden_dim
#             self.feature_layers.append(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU()))

#         # Projection to scalar S and expansion back to vector
#         self.to_S = nn.Linear(hidden_dim, 1)
#         self.expand_S = nn.Linear(1, hidden_dim)

#         # Decoder: input dimension depends on residual_type
#         dec_in = hidden_dim * 2 if residual_type == "concat" else hidden_dim
#         decoder_layers = []
#         for _ in range(num_decoder_layers - 1):
#             decoder_layers.append(nn.Linear(dec_in, hidden_dim))
#             decoder_layers.append(nn.ReLU())
#             dec_in = hidden_dim
#         decoder_layers.append(nn.Linear(dec_in, 1))
#         decoder_layers.append(nn.Sigmoid())
#         self.decoder = nn.Sequential(*decoder_layers)

#         # Loss functions and optimizer
#         self.loss_z = nn.MSELoss()  # MSE(S_pred, z)
#         self.loss_y = nn.BCELoss()  # BCE(y_pred, y)
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)

#     def forward(self, x: torch.Tensor):
#         """
#         Forward pass:
#           - Encode x through encoder layers, collect residuals.
#           - Compute scalar S from final encoder output, then expand to vector.
#           - Combine expanded S with last encoder latent.
#           - Decode to y_hat.
#         Returns:
#             S (Tensor [batch,1]), y_hat (Tensor [batch,1])
#         """
#         residuals = []
#         h = x
#         for layer in self.feature_layers:
#             h = layer(h)
#             residuals.append(h)

#         # Compute S and expanded S
#         S = self.to_S(h)            # [batch,1]
#         exp_S = self.expand_S(S)    # [batch,hidden_dim]

#         # Combine expanded S with last encoder output
#         last = residuals[-1]        # [batch,hidden_dim]
#         if self.residual_type == "sum":
#             combined = exp_S + last
#         else:  # "concat"
#             combined = torch.cat([exp_S, last], dim=1)

#         y_hat = self.decoder(combined)  # [batch,1]
#         return S, y_hat

#     def train_model(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         z: torch.Tensor,
#         x_val: torch.Tensor = None,
#         y_val: torch.Tensor = None,
#         z_val: torch.Tensor = None,
#         early_stopping: int = 10,
#         record_loss: bool = False,
#     ):
#         """
#         Joint training with combined loss: MSE(S_pred, z) + BCE(y_pred, y).
#         Supports optional validation and early stopping on BCE.

#         Args:
#             x (Tensor): Training features [N, D].
#             y (Tensor): Main labels [N, 1].
#             z (Tensor): Privileged side info [N, 1].
#             x_val, y_val, z_val (Tensor, optional): Validation sets.
#             early_stopping (int): Patience for early stopping on validation BCE loss.
#             record_loss (bool): If True, returns (train_history, val_history).
#         Returns:
#             (train_hist, val_hist) if record_loss else None
#         """
#         train_hist, val_hist = [], []
#         best_val = float("inf")
#         patience = 0

#         for epoch in range(self.epochs):
#             self.train()
#             self.optimizer.zero_grad()
#             S_pred, y_pred = self.forward(x)
#             loss = self.loss_z(S_pred, z) + self.loss_y(y_pred, y)
#             loss.backward()
#             self.optimizer.step()

#             if record_loss:
#                 train_hist.append(self.loss_y(y_pred, y).item())

#             # Validation step
#             if x_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     _, yv_pred = self.forward(x_val)
#                     val_loss = self.loss_y(yv_pred, y_val).item()
#                 if record_loss:
#                     val_hist.append(val_loss)
#                 if val_loss < best_val:
#                     best_val = val_loss
#                     patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping:
#                         break

#         return (train_hist, val_hist) if record_loss else None

#     def evaluate(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         Evaluate on test set: compute accuracy, AUC, BCE loss,
#         and per-class precision/recall/F1.

#         Returns:
#             Dict with keys: "accuracy", "auc", "bce_loss", 
#             "p0", "r0", "f0", "p1", "r1", "f1"
#         """
#         self.eval()
#         with torch.no_grad():
#             _, y_pred = self.forward(x)
#             y_pred_flat = y_pred.view(-1)
#             y_flat = y.view(-1)
#             y_hat = (y_pred_flat >= 0.5).float()

#             # Accuracy
#             accuracy = (y_hat == y_flat).float().mean().item()

#             # AUC
#             try:
#                 auc_score = roc_auc_score(y_flat.cpu().numpy(), y_pred_flat.cpu().numpy())
#             except ValueError:
#                 auc_score = 0.0

#             bce_loss = self.loss_y(y_pred_flat, y_flat).item()
#             p, r, f, _ = precision_recall_fscore_support(
#                 y_flat.cpu().numpy(), y_hat.cpu().numpy(), zero_division=0
#             )

#         return {
#             "accuracy": accuracy,
#             "auc": auc_score,
#             "bce_loss": bce_loss,
#             "p0": p[0],
#             "r0": r[0],
#             "f0": f[0],
#             "p1": p[1],
#             "r1": r[1],
#             "f1": f[1],
#         }


# # =========================
# #     UTILITY FUNCTIONS
# # =========================

# def plot_loss_curve(train_losses, val_losses, filename: str):
#     """
#     Plot and save training vs. validation BCE loss curve.
#     """
#     plt.figure(figsize=(8, 6))
#     plt.plot(train_losses, label="Train BCE Loss")
#     plt.plot(val_losses, label="Validation BCE Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("BCE Loss")
#     plt.title("Train vs Validation BCE Loss")
#     plt.legend()
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     plt.savefig(filename)
#     plt.close()


# def grid_search_direct(
#     model_class,
#     param_grid: dict,
#     X: torch.Tensor,
#     Y: torch.Tensor,
#     Z: torch.Tensor,
#     cv: int = 3,
#     early_stopping: int = 10,
# ):
#     """
#     Perform grid search for Direct-pattern models using ROC AUC on the main task.

#     Args:
#         model_class: Class inheriting from nn.Module, expects __init__(input_dim, **params)
#                      and train_model() / evaluate() methods.
#         param_grid: Dict where keys are hyperparameter names and values are lists of candidates.
#         X: Tensor of shape (N, D), features for grid search.
#         Y: Tensor of shape (N, 1), main labels.
#         Z: Tensor of shape (N, 1), side-information labels.
#         cv: Number of cross-validation folds.
#         early_stopping: Patience for early stopping in train_model.

#     Returns:
#         best_params: Dict of hyperparameters with highest mean AUC.
#         best_auc: Float, corresponding average AUC.
#     """
#     best_auc = -np.inf
#     best_params = None
#     keys = list(param_grid.keys())

#     for combo in itertools.product(*(param_grid[k] for k in keys)):
#         params = dict(zip(keys, combo))
#         aucs = []
#         kf = KFold(n_splits=cv, shuffle=True, random_state=42)

#         for train_idx, val_idx in kf.split(X):
#             x_tr, x_va = X[train_idx], X[val_idx]
#             y_tr, y_va = Y[train_idx], Y[val_idx]
#             z_tr, z_va = Z[train_idx], Z[val_idx]

#             # Instantiate model with current hyperparameters
#             model = model_class(input_dim=X.shape[1], **params)

#             # Train the model on training fold
#             model.train_model(
#                 x_tr,
#                 y_tr,
#                 z_tr,
#                 x_val=x_va,
#                 y_val=y_va,
#                 z_val=z_va,
#                 early_stopping=early_stopping,
#                 record_loss=False,
#             )

#             # Evaluate on validation fold (main task AUC)
#             with torch.no_grad():
#                 _, yv_pred = model.forward(x_va)
#                 try:
#                     fold_auc = roc_auc_score(y_va.cpu().numpy(), yv_pred.detach().cpu().numpy())
#                 except ValueError:
#                     fold_auc = 0.0
#                 aucs.append(fold_auc)

#         mean_auc = float(np.mean(aucs))
#         if mean_auc > best_auc:
#             best_auc = mean_auc
#             best_params = params.copy()

#     return best_params, best_auc


# def run_experiments_direct(
#     model_class,
#     best_params: dict,
#     train_x: torch.Tensor,
#     train_y: torch.Tensor,
#     train_z: torch.Tensor,
#     test_x: torch.Tensor,
#     test_y: torch.Tensor,
#     num_runs: int = 10,
#     sample_size_per_class: int = 250,
#     early_stopping: int = 10,
# ):
#     """
#     Execute multiple random-sampled experiments for a Direct-pattern model.

#     For each run:
#       - Balanced sampling of sample_size_per_class positives and negatives from train pool.
#       - 80/20 train/validation split.
#       - Train with record_loss=True to save loss curves.
#       - Evaluate on a random 125-subset of test pool.
#       - Return metrics per run.

#     Args:
#         model_class: Direct-pattern model class.
#         best_params: Hyperparameters from grid search.
#         train_x, train_y, train_z: Tensors for training pool.
#         test_x, test_y: Tensors for test pool.
#         num_runs: Number of repeated experiments.
#         sample_size_per_class: Number of examples per class to sample.
#         early_stopping: Patience for early stopping.

#     Returns:
#         metrics_list: List of dicts from model.evaluate() across runs.
#     """
#     metrics_list = []

#     # Build DataFrame to identify indices for each class
#     train_df = pd.DataFrame(train_x.numpy())
#     train_df["Diabetes_binary"] = train_y.numpy().flatten()
#     class1_indices = train_df[train_df["Diabetes_binary"] == 1].index.tolist()
#     class0_indices = train_df[train_df["Diabetes_binary"] == 0].index.tolist()
#     test_pool_size = len(test_x)

#     for run_idx in range(num_runs):
#         # 1) Balanced sampling
#         sampled1 = random.sample(class1_indices, min(len(class1_indices), sample_size_per_class))
#         sampled0 = random.sample(class0_indices, min(len(class0_indices), sample_size_per_class))
#         train_idx = sampled1 + sampled0

#         x_sel = train_x[train_idx]
#         y_sel = train_y[train_idx]
#         z_sel = train_z[train_idx]

#         # 2) Split into train/validation (80/20)
#         perm = np.random.permutation(len(train_idx))
#         split = int(0.8 * len(train_idx))
#         tr_idx, va_idx = perm[:split], perm[split:]
#         x_tr, y_tr, z_tr = x_sel[tr_idx], y_sel[tr_idx], z_sel[tr_idx]
#         x_va, y_va, z_va = x_sel[va_idx], y_sel[va_idx], z_sel[va_idx]

#         # 3) Sample random 125 examples from test pool
#         test_idx = random.sample(range(test_pool_size), min(125, test_pool_size))
#         x_test_run = test_x[test_idx]
#         y_test_run = test_y[test_idx]

#         # 4) Instantiate model with best_params
#         model = model_class(input_dim=train_x.shape[1], **best_params)

#         # 5) Train model with recording loss
#         train_losses, val_losses = model.train_model(
#             x_tr,
#             y_tr,
#             z_tr,
#             x_val=x_va,
#             y_val=y_va,
#             z_val=z_va,
#             early_stopping=early_stopping,
#             record_loss=True,
#         )

#         # 6) Plot loss curve
#         plot_loss_curve(train_losses, val_losses, f"img/{model_class.__name__}_run{run_idx+1}.png")

#         # 7) Evaluate on test subset
#         metrics = model.evaluate(x_test_run, y_test_run)
#         metrics_list.append(metrics)
#         print(f"Run {run_idx+1} metrics:", metrics)

#     return metrics_list


# def aggregate_metrics(metrics_list: list):
#     """
#     Aggregate a list of metric dictionaries by computing mean ± std for each key.
#     """
#     agg = {}
#     for key in metrics_list[0].keys():
#         vals = [m[key] for m in metrics_list]
#         mean_val = np.mean(vals)
#         std_val = np.std(vals)
#         agg[key] = f"{mean_val:.4f} ± {std_val:.4f}"
#     return agg


# def write_results_csv(filename: str, method: str, metrics: dict, params: str):
#     """
#     Append a row of results (metrics + params) to a CSV file.
#     """
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     with open(filename, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(
#             [
#                 method,
#                 metrics.get("auc", ""),
#                 metrics.get("accuracy", ""),
#                 metrics.get("p0", ""),
#                 metrics.get("r0", ""),
#                 metrics.get("f0", ""),
#                 metrics.get("p1", ""),
#                 metrics.get("r1", ""),
#                 metrics.get("f1", ""),
#                 metrics.get("bce_loss", ""),
#                 params,
#             ]
#         )


# def print_aggregated_results(method: str, metrics: dict, params: str):
#     """
#     Print aggregated results in a readable format.
#     """
#     print("\n=== Final Aggregated Results ===")
#     print(f"Method: {method}")
#     print(f"AUC: {metrics.get('auc','')}")
#     print(f"Accuracy: {metrics.get('accuracy','')}")
#     print(f"Class0 → Precision: {metrics.get('p0','')}, Recall: {metrics.get('r0','')}, F1: {metrics.get('f0','')}")
#     print(f"Class1 → Precision: {metrics.get('p1','')}, Recall: {metrics.get('r1','')}, F1: {metrics.get('f1','')}")
#     print(f"BCE Loss: {metrics.get('bce_loss','')}")
#     print(f"Best Parameters: {params}")
#     print("---------------------------------------------------")


# # =========================
# #          MAIN
# # =========================

# def main():
#     """
#     Main entry point to run Direct-pattern experiments.
#     Loads data, performs grid search, runs repeated experiments, 
#     aggregates, and saves/prints results.
#     """
#     # Ensure reproducibility
#     set_seed(42)

#     OUTPUT_CSV = "final_results.csv"

#     # Initialize CSV with header if not exists
#     dir_path = os.path.dirname(OUTPUT_CSV)
#     print(f"DEBUG: dir_path = '{dir_path}'")  # Optional

#     if dir_path and not os.path.exists(dir_path):
#         os.makedirs(dir_path, exist_ok=True)

#     if not os.path.exists(OUTPUT_CSV):
#         with open(OUTPUT_CSV, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(
#                 [
#                     "Method",
#                     "AUC",
#                     "Overall_Acc",
#                     "Acc_Class0",
#                     "Prec_Class0",
#                     "Rec_Class0",
#                     "F1_Class0",
#                     "Acc_Class1",
#                     "Prec_Class1",
#                     "Rec_Class1",
#                     "F1_Class1",
#                     "BCE_Loss",
#                     "Params",
#                 ]
#             )

#     # 2) Preprocess data
#     dp = DirectDataPreprocessor(dataset_id=891, side_info_path="prompting/augmented_data.csv")
#     grid_x, grid_y, grid_z, train_x, train_y, train_z, test_x, test_y = dp.preprocess()
#     # print("70B")
#     # 3) Define experiment configurations for each Direct model
#     experiments = [
#         {
#             "name": "Simul_NoDecomp",
#             "model": DirectPatternNoDecomp,
#             "param_grid": {
#                 "hidden_dim": [64, 128, 256],
#                 "num_layers": [1, 2, 3, 4],
#                 "lr": [0.001, 0.01],
#                 "epochs": [100],  # fix epochs for grid search
#                 "residual_type": ["concat"],  # fix residual_type for grid search
#             },
#         },
#         # {
#         #     "name": "Simul_with_Decomp",
#         #     "model": DirectPatternResidual,
#         #     "param_grid": {
#         #         "hidden_dim": [64, 128, 256],
#         #         "num_layers": [1, 2, 3, 4],
#         #         "lr": [0.001, 0.01],
#         #         "epochs": [100],
#         #         "residual_type": ["concat"],
#         #     }
#         # },
#         # {
#         #     "name": "Decoupled_NoDecomp",
#         #     "model": DirectDecoupledResidualNoDecomp,
#         #     "param_grid": {
#         #         "hidden_dim": [64, 128, 256],
#         #         "num_layers": [1, 2, 3, 4],
#         #         "lr": [0.001, 0.01],
#         #         "epochs": [100],
#         #         "residual_type": ["concat"],
#         #     },
#         # },
#         # {
#         #     "name": "Decoupled_with_Decomp",
#         #     "model": DirectDecoupledResidualModel,
#         #     "param_grid": {
#         #         "hidden_dim": [64, 128, 256],
#         #         "num_layers": [1, 2, 3, 4],
#         #         "lr": [0.001, 0.01],
#         #         "epochs": [100],
#         #         "residual_type": ["concat"],
#         #     },
#         # },
#     ]

#     # 4) Loop over each experiment configuration
#     for cfg in experiments:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#         name = cfg["name"]
#         ModelClass = cfg["model"]
#         param_grid = cfg["param_grid"]

#         print(f"\n=== Grid Search for {name} ===")
#         # 4a) Hyperparameter grid search (3-fold CV)
#         best_params, best_auc = grid_search_direct(
#             ModelClass, param_grid, grid_x, grid_y, grid_z, cv=3, early_stopping=10
#         )
#         print(f"[{name}] Best params: {best_params}, CV AUC: {best_auc:.4f}")

#         # 4b) Final repeated experiments with best_params
#         print(f"\n=== Running Repeated Experiments for {name} ===")
#         results = run_experiments_direct(
#             ModelClass,
#             best_params,
#             train_x,
#             train_y,
#             train_z,
#             test_x,
#             test_y,
#             num_runs=10,
#             sample_size_per_class=250,
#             early_stopping=10,
#         )

#         # 4c) Aggregate metrics
#         aggregated = aggregate_metrics(results)

#         # 4d) Save to CSV and print
#         # write_results_csv(OUTPUT_CSV, name, aggregated, str(best_params))
#         print_aggregated_results(name, aggregated, str(best_params))


# if __name__ == "__main__":
#     main()


# direct_experiments.py

import os
import csv
import random
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from ucimlrepo import fetch_ucirepo


# =========================
#    GLOBAL SEED FUNCTION
# =========================

def set_seed(seed: int):
    """
    Set random seed for reproducibility across random, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
#   DATA PREPROCESSOR
# =========================

class DirectDataPreprocessor:
    """
    Preprocessor for Direct-pattern models.
    Uses LLM-augmented side information (privileged) during training,
    but excludes it from the test set.
    """

    def __init__(
        self,
        dataset_id: int = 891,
        side_info_path: str = "prompting/augmented_data.csv",
    ):
        """
        Args:
            dataset_id (int): UCI repository dataset ID (default: 891).
            side_info_path (str): Path to CSV containing LLM-augmented side information.
        """
        # Fetch original UCI dataset
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features  # pandas DataFrame of features
        self.y_original = self.original_data.data.targets   # pandas Series of main labels

        # StandardScaler for continuous features (BMI and side info)
        self.scaler = StandardScaler()

        # Store path to LLM-generated side information CSV
        self.side_path = side_info_path

        # Columns used as input features
        self.training_cols = [
            "HighBP",
            "HighChol",
            "CholCheck",
            "BMI",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "DiffWalk",
            "Sex",
            "Age",
            "Education",
            "Income",
        ]
        # Column names for main target and privileged side information
        self.target = "Diabetes_binary"
        self.side_info = "predict_hba1c"  # LLM-generated privileged information

    def preprocess(self):
        """
        Execute preprocessing and return tensors for:
          - grid set (balanced) for hyperparameter tuning
          - training pool (with side information)
          - test set (excluded from side information)

        Returns:
            grid_x, grid_y, grid_z, train_x, train_y, train_z, test_x, test_y
        """
        continuous_cols = ["BMI", self.side_info]

        # 1) Load augmented CSV (contains side information)
        augmented_df = pd.read_csv(self.side_path)
        augmented_df = augmented_df[self.training_cols + [self.target, self.side_info]]

        # 2) Drop rows with missing main target or side information
        augmented_df = augmented_df.dropna(subset=[self.target, self.side_info])

        # 3) Create a balanced 'grid' set for hyperparameter tuning
        pos_idx = augmented_df[augmented_df[self.target] == 1].index.tolist()
        neg_idx = augmented_df[augmented_df[self.target] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
        grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
        grid_df = pd.concat([grid_pos, grid_neg])

        # The remaining data after removing grid_df is used as training pool
        train_pool = augmented_df.drop(index=grid_df.index)

        # 4) Construct a separate test set using original UCI data (no side information)
        original_df = self.X_original.copy()
        original_df[self.target] = self.y_original
        # Drop any rows that appear in augmented_df to avoid overlap
        test_pool = original_df.drop(index=augmented_df.index, errors="ignore")
        # Keep only feature columns and main target
        test_pool = test_pool[self.training_cols + [self.target]]

        # 5) Fit scaler on BMI and side_info from training pool, then transform grid and train_pool
        self.scaler.fit(train_pool[continuous_cols])
        grid_df_scaled = grid_df.copy()
        grid_df_scaled[continuous_cols] = self.scaler.transform(grid_df[continuous_cols])
        train_pool_scaled = train_pool.copy()
        train_pool_scaled[continuous_cols] = self.scaler.transform(train_pool[continuous_cols])

        # 6) For test set, only normalize BMI using scaler parameters (side_info not available)
        bmi_mean = self.scaler.mean_[0]
        bmi_scale = self.scaler.scale_[0]
        test_pool = test_pool.copy()
        test_pool["BMI"] = (test_pool["BMI"] - bmi_mean) / bmi_scale

        # 7) Convert DataFrames to PyTorch tensors
        def to_tensor(df: pd.DataFrame, multi: bool = False):
            """
            Convert features and targets (and side info if multi=True) to torch.Tensor.
            """
            x = torch.tensor(df[self.training_cols].values, dtype=torch.float32)
            y = torch.tensor(df[self.target].values, dtype=torch.float32).view(-1, 1)
            if multi:
                z = torch.tensor(df[self.side_info].values, dtype=torch.float32).view(-1, 1)
                return x, y, z
            else:
                return x, y

        # Tensors for grid set
        grid_x, grid_y, grid_z = to_tensor(grid_df_scaled, multi=True)
        # Tensors for training pool
        train_x, train_y, train_z = to_tensor(train_pool_scaled, multi=True)
        # Tensors for test set (no side_info)
        test_x, test_y = to_tensor(test_pool, multi=False)

        return grid_x, grid_y, grid_z, train_x, train_y, train_z, test_x, test_y


# =========================
#      MODEL CLASSES
# =========================

class DirectDecoupledResidualNoDecomp(nn.Module):
    """
    Decoupled Residual Model WITHOUT explicit expansion, 
    with early stopping in fine-tune phase.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_decoder_layers: int = 1,
        residual_type: str = "concat",  # either "concat" or "sum"
        lr: float = 0.001,
        epochs: int = 100,
    ):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension size.
            num_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            residual_type (str): "concat" or "sum" for combining S and residual.
            lr (float): Learning rate for both phases.
            epochs (int): Number of epochs per phase.
        """
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.residual_type = residual_type

        # Encoder: list of (Linear -> ReLU) modules
        self.encoder_layers = nn.ModuleList()
        dim = input_dim
        for _ in range(num_layers):
            self.encoder_layers.append(nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU()))
            dim = hidden_dim

        # Learnable weights for residual combination
        self.alpha = nn.Parameter(torch.ones(num_layers))

        # Linear projection of final encoder output to scalar S
        self.to_S = nn.Linear(hidden_dim, 1)

        # Decoder network: input dimension depends on residual_type
        dec_in = hidden_dim + 1 if residual_type == "concat" else hidden_dim
        decoder_layers = []
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(dec_in, hidden_dim))
            decoder_layers.append(nn.ReLU())
            dec_in = hidden_dim
        decoder_layers.append(nn.Linear(dec_in, 1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Loss functions
        self.loss_z = nn.MSELoss()  # for pretrain (MSE between S and privileged z)
        self.loss_y = nn.BCELoss()  # for fine-tune (BCE between y_hat and y)

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
          - Compute encoder outputs for each layer.
          - Compute weighted sum of all encoder outputs as 'residual'.
          - Project last encoder output to scalar S via to_S.
          - Combine S and residual (sum or concat).
          - Pass through decoder to get y_hat.
        Returns:
            S (Tensor [batch,1]), y_hat (Tensor [batch,1])
        """
        outs = []
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
            outs.append(h)

        # Softmax over alpha to weight each encoder output
        w = torch.softmax(self.alpha, dim=0)
        residual = sum(wi * oi for wi, oi in zip(w, outs))  # [batch, hidden_dim]

        # Compute scalar S from the last encoder output
        S = self.to_S(outs[-1])  # [batch, 1]

        # Combine S and residual
        if self.residual_type == "sum":
            combined = S + residual  # broadcasting: [batch, hidden_dim]
        else:  # "concat"
            combined = torch.cat([S, residual], dim=1)  # [batch, hidden_dim+1]

        # Decode to y_hat
        y_hat = self.decoder(combined)  # [batch, 1]
        return S, y_hat

    def pretrain_phase(self, x: torch.Tensor, z: torch.Tensor, epochs: int = None, lr: float = None):
        """
        Phase 1: Pre-train encoder layers, to_S, and alpha by minimizing MSE(S, z).
        """
        e = epochs or self.epochs
        lr0 = lr or self.lr
        optimizer = optim.Adam(
            list(self.encoder_layers.parameters()) + list(self.to_S.parameters()) + [self.alpha],
            lr=lr0,
        )
        self.train()
        for _ in range(e):
            optimizer.zero_grad()
            S_pred, _ = self.forward(x)
            loss = self.loss_z(S_pred, z)
            loss.backward()
            optimizer.step()

    def train_model(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        x_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        z_val: torch.Tensor = None,
        early_stopping: int = 10,
        record_loss: bool = False,
    ):
        """
        Full training: 
          - Phase 1: Pre-train encoder + S projection.
          - Phase 2: Freeze encoder components, train decoder with early stopping on BCE.

        Args:
            x (Tensor): Training features [N, D].
            y (Tensor): Main labels [N, 1].
            z (Tensor): Privileged side info [N, 1].
            x_val, y_val, z_val (Tensor, optional): Validation sets.
            early_stopping (int): Patience for early stopping on validation BCE loss.
            record_loss (bool): If True, returns (train_history, val_history).
        Returns:
            (train_hist, val_hist) if record_loss else None
        """
        # Phase 1: pre-train
        self.pretrain_phase(x, z)

        # Freeze encoder components and S projection
        for p in self.encoder_layers.parameters():
            p.requires_grad = False
        for p in self.to_S.parameters():
            p.requires_grad = False
        self.alpha.requires_grad = False

        # Phase 2: fine-tune decoder only
        optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        train_hist, val_hist = [], []
        best_val = float("inf")
        patience = 0

        for epoch in range(self.epochs):
            # Training step on decoder
            self.train()
            optimizer.zero_grad()
            _, y_pred = self.forward(x)  # [N,1]
            loss_train = self.loss_y(y_pred, y)
            loss_train.backward()
            optimizer.step()

            if record_loss:
                train_hist.append(loss_train.item())

            # Validation step for early stopping
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    _, yv_pred = self.forward(x_val)
                    loss_val = self.loss_y(yv_pred, y_val)
                if record_loss:
                    val_hist.append(loss_val.item())
                if loss_val.item() < best_val:
                    best_val = loss_val.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping:
                        break

        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, x: torch.Tensor, y: torch.Tensor):
        """
        Evaluate on test set: compute AUC, BCE loss, and per-class precision/recall/F1.

        Returns:
            Dict with keys: "auc", "bce_loss", "p0", "r0", "f0", "p1", "r1", "f1"
        """
        self.eval()
        with torch.no_grad():
            _, y_pred = self.forward(x)  # [N,1]
            y_pred_flat = y_pred.view(-1)
            y_flat = y.view(-1)
            preds = (y_pred_flat >= 0.5).float()

            # Compute AUC (if possible)
            try:
                auc_score = roc_auc_score(y_flat.cpu().numpy(), y_pred_flat.cpu().numpy())
            except ValueError:
                auc_score = 0.0

            bce_loss = self.loss_y(y_pred_flat, y_flat).item()
            p, r, f, _ = precision_recall_fscore_support(
                y_flat.cpu().numpy(), preds.cpu().numpy(), zero_division=0
            )

        return {
            "auc": auc_score,
            "bce_loss": bce_loss,
            "p0": p[0],
            "r0": r[0],
            "f0": f[0],
            "p1": p[1],
            "r1": r[1],
            "f1": f[1],
        }


class DirectDecoupledResidualModel(nn.Module):
    """
    Decoupled Residual Model WITH explicit expansion,
    with early stopping in fine-tune phase.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_decoder_layers: int = 1,
        residual_type: str = "concat",  # "concat" or "sum"
        lr: float = 0.001,
        epochs: int = 100,
    ):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension size.
            num_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            residual_type (str): "concat" or "sum".
            lr (float): Learning rate for both phases.
            epochs (int): Number of epochs per phase.
        """
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.residual_type = residual_type

        # Encoder: list of (Linear -> ReLU) modules
        self.encoder_layers = nn.ModuleList()
        dim = input_dim
        for _ in range(num_layers):
            self.encoder_layers.append(nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU()))
            dim = hidden_dim

        # Learnable weights for residual combination
        self.alpha = nn.Parameter(torch.ones(num_layers))

        # Projection to scalar S and expansion back to hidden_dim
        self.to_S = nn.Linear(hidden_dim, 1)
        self.expand_S = nn.Linear(1, hidden_dim)

        # Decoder network: input dimension depends on residual_type
        dec_in = hidden_dim * 2 if residual_type == "concat" else hidden_dim
        decoder_layers = []
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(dec_in, hidden_dim))
            decoder_layers.append(nn.ReLU())
            dec_in = hidden_dim
        decoder_layers.append(nn.Linear(dec_in, 1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Loss functions
        self.loss_z = nn.MSELoss()  # for pretrain
        self.loss_y = nn.BCELoss()  # for fine-tune

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
          - Compute encoder outputs for each layer.
          - Compute weighted sum of encoder outputs as 'residual'.
          - Project last encoder output to S, then expand to hidden_dim.
          - Combine expanded S with residual.
          - Decode to y_hat.
        Returns:
            S (Tensor [batch,1]), y_hat (Tensor [batch,1])
        """
        outs = []
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
            outs.append(h)

        # Weighted residual
        w = torch.softmax(self.alpha, dim=0)
        residual = sum(wi * oi for wi, oi in zip(w, outs))  # [batch, hidden_dim]

        # Project to scalar S and expand back
        S = self.to_S(outs[-1])            # [batch, 1]
        exp_S = self.expand_S(S)           # [batch, hidden_dim]

        # Combine expanded S and residual
        if self.residual_type == "sum":
            combined = exp_S + residual  # [batch, hidden_dim]
        else:  # "concat"
            combined = torch.cat([exp_S, residual], dim=1)  # [batch, hidden_dim*2]

        # Decode to y_hat
        y_hat = self.decoder(combined)  # [batch, 1]
        return S, y_hat

    def pretrain_phase(self, x: torch.Tensor, z: torch.Tensor, epochs: int = None, lr: float = None):
        """
        Phase 1: Train encoder, to_S, expand_S, and alpha to minimize MSE(S, z).
        """
        e = epochs or self.epochs
        lr0 = lr or self.lr
        optimizer = optim.Adam(
            list(self.encoder_layers.parameters())
            + list(self.to_S.parameters())
            + list(self.expand_S.parameters())
            + [self.alpha],
            lr=lr0,
        )
        self.train()
        for _ in range(e):
            optimizer.zero_grad()
            S_pred, _ = self.forward(x)
            loss = self.loss_z(S_pred, z)
            loss.backward()
            optimizer.step()

    def train_model(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        x_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        z_val: torch.Tensor = None,
        early_stopping: int = 10,
        record_loss: bool = False,
    ):
        """
        Full training:
          - Phase 1: Pretrain encoder + to_S + expand_S + alpha.
          - Phase 2: Freeze encoder components, train decoder with early stopping on BCE.

        Args:
            x (Tensor): Training features [N, D].
            y (Tensor): Main labels [N, 1].
            z (Tensor): Privileged side info [N, 1].
            x_val, y_val, z_val (Tensor, optional): Validation sets.
            early_stopping (int): Patience for early stopping on validation BCE loss.
            record_loss (bool): If True, returns (train_history, val_history).
        Returns:
            (train_hist, val_hist) if record_loss else None
        """
        # Phase 1: pretrain
        self.pretrain_phase(x, z)

        # Freeze encoder, to_S, expand_S, and alpha
        for p in self.encoder_layers.parameters():
            p.requires_grad = False
        for p in self.to_S.parameters():
            p.requires_grad = False
        for p in self.expand_S.parameters():
            p.requires_grad = False
        self.alpha.requires_grad = False

        # Phase 2: train decoder
        optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        train_hist, val_hist = [], []
        best_val = float("inf")
        patience = 0

        for epoch in range(self.epochs):
            # Training step
            self.train()
            optimizer.zero_grad()
            _, y_pred = self.forward(x)
            loss_train = self.loss_y(y_pred, y)
            loss_train.backward()
            optimizer.step()

            if record_loss:
                train_hist.append(loss_train.item())

            # Validation step
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    _, yv_pred = self.forward(x_val)
                    loss_val = self.loss_y(yv_pred, y_val)
                if record_loss:
                    val_hist.append(loss_val.item())
                if loss_val.item() < best_val:
                    best_val = loss_val.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping:
                        break

        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, x: torch.Tensor, y: torch.Tensor):
        """
        Evaluate on test set: compute AUC, BCE loss, and per-class precision/recall/F1.

        Returns:
            Dict with keys: "auc", "bce_loss", "p0", "r0", "f0", "p1", "r1", "f1"
        """
        self.eval()
        with torch.no_grad():
            _, y_pred = self.forward(x)
            y_pred_flat = y_pred.view(-1)
            y_flat = y.view(-1)
            preds = (y_pred_flat >= 0.5).float()

            # Compute AUC
            try:
                auc_score = roc_auc_score(y_flat.cpu().numpy(), y_pred_flat.cpu().numpy())
            except ValueError:
                auc_score = 0.0

            bce_loss = self.loss_y(y_pred_flat, y_flat).item()
            p, r, f, _ = precision_recall_fscore_support(
                y_flat.cpu().numpy(), preds.cpu().numpy(), zero_division=0
            )

        return {
            "auc": auc_score,
            "bce_loss": bce_loss,
            "p0": p[0],
            "r0": r[0],
            "f0": f[0],
            "p1": p[1],
            "r1": r[1],
            "f1": f[1],
        }


class DirectPatternNoDecomp(nn.Module):
    """
    Direct Pattern Model WITHOUT explicit expansion layer.

    - Encoder extracts features.
    - to_S compresses to scalar S.
    - Combine S with weighted sum of encoder outputs (residual).
    - Decoder predicts y_hat.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_decoder_layers: int = 1,
        residual_type: str = "concat",  # "concat" or "sum"
        lr: float = 0.001,
        epochs: int = 100,
    ):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension size.
            num_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            residual_type (str): "concat" or "sum".
            lr (float): Learning rate for joint training.
            epochs (int): Number of epochs for joint training.
        """
        super().__init__()
        self.epochs = epochs
        self.residual_type = residual_type

        # Encoder: list of (Linear -> ReLU) modules
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.feature_layers.append(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU()))

        # Learnable weights for residual combination
        self.alpha = nn.Parameter(torch.ones(num_layers))

        # Projection to scalar S
        self.to_S = nn.Linear(hidden_dim, 1)

        # Decoder: input dim depends on residual_type
        decoder_input = hidden_dim + 1 if residual_type == "concat" else hidden_dim
        decoder_layers = []
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(decoder_input, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_input = hidden_dim
        decoder_layers.append(nn.Linear(decoder_input, 1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Loss functions
        self.loss_z = nn.MSELoss()  # MSE(S_pred, z)
        self.loss_y = nn.BCELoss()  # BCE(y_pred, y)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
          - Compute encoder outputs for each layer.
          - Compute weighted sum of encoder outputs as 'residual'.
          - Compute scalar S from last encoder output.
          - Combine S and residual (sum or concat).
          - Decode to y_hat.
        Returns:
            S (Tensor [batch,1]), y_hat (Tensor [batch,1])
        """
        outputs = []
        h = x
        for layer in self.feature_layers:
            h = layer(h)
            outputs.append(h)

        weights = torch.softmax(self.alpha, dim=0)
        residual = sum(w * o for w, o in zip(weights, outputs))  # [batch, hidden_dim]

        S = self.to_S(outputs[-1])  # [batch, 1]

        if self.residual_type == "sum":
            combined = S + residual  # [batch, hidden_dim]
        else:  # "concat"
            combined = torch.cat([S, residual], dim=1)  # [batch, hidden_dim+1]

        y_pred = self.decoder(combined)  # [batch, 1]
        return S, y_pred

    def train_model(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        x_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        z_val: torch.Tensor = None,
        early_stopping: int = 10,
        record_loss: bool = False,
    ):
        """
        Joint training with combined loss: MSE(S_pred, z) + BCE(y_pred, y).
        Supports optional validation and early stopping on BCE.

        Args:
            x (Tensor): Training features [N, D].
            y (Tensor): Main labels [N, 1].
            z (Tensor): Privileged side info [N, 1].
            x_val, y_val, z_val (Tensor, optional): Validation sets.
            early_stopping (int): Patience for early stopping on validation BCE loss.
            record_loss (bool): If True, returns (train_history, val_history).
        Returns:
            (train_hist, val_hist) if record_loss else None
        """
        train_hist, val_hist = [], []
        best_val = float("inf")
        patience = 0

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            S_pred, y_pred = self.forward(x)
            loss = self.loss_z(S_pred, z) + self.loss_y(y_pred, y)
            loss.backward()
            self.optimizer.step()

            if record_loss:
                train_hist.append(self.loss_y(y_pred, y).item())

            # Validation step
            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    _, yv_pred = self.forward(x_val)
                    val_loss = self.loss_y(yv_pred, y_val).item()
                if record_loss:
                    val_hist.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping:
                        break

        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, x: torch.Tensor, y: torch.Tensor):
        """
        Evaluate on test set: compute accuracy, AUC, BCE loss, 
        and per-class precision/recall/F1.

        Returns:
            Dict with keys: "accuracy", "auc", "bce_loss", 
            "p0", "r0", "f0", "p1", "r1", "f1"
        """
        self.eval()
        with torch.no_grad():
            _, y_pred = self.forward(x)  # [N,1]
            y_pred_flat = y_pred.view(-1)
            y_flat = y.view(-1)
            y_hat = (y_pred_flat >= 0.5).float()

            # Accuracy
            accuracy = (y_hat == y_flat).float().mean().item()

            # AUC
            try:
                auc_score = roc_auc_score(y_flat.cpu().numpy(), y_pred_flat.cpu().numpy())
            except ValueError:
                auc_score = 0.0

            bce_loss = self.loss_y(y_pred_flat, y_flat).item()
            p, r, f, _ = precision_recall_fscore_support(
                y_flat.cpu().numpy(), y_hat.cpu().numpy(), zero_division=0
            )

        return {
            "accuracy": accuracy,
            "auc": auc_score,
            "bce_loss": bce_loss,
            "p0": p[0],
            "r0": r[0],
            "f0": f[0],
            "p1": p[1],
            "r1": r[1],
            "f1": f[1],
        }


class DirectPatternResidual(nn.Module):
    """
    Direct Pattern Model WITH explicit expansion layer.

    - Encoder compresses X to latent h.
    - to_S compresses h to scalar S.
    - expand_S expands S back to hidden dimension.
    - Combine expanded S + last encoder output (sum or concat).
    - Decoder predicts y_hat.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_decoder_layers: int = 1,
        residual_type: str = "concat",  # "concat" or "sum"
        lr: float = 0.001,
        epochs: int = 300,
    ):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Hidden dimension size.
            num_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            residual_type (str): "concat" or "sum".
            lr (float): Learning rate for joint training.
            epochs (int): Number of epochs for joint training.
        """
        super().__init__()
        self.epochs = epochs
        self.residual_type = residual_type

        # Build encoder layers
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.feature_layers.append(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU()))

        # Projection to scalar S and expansion back to vector
        self.to_S = nn.Linear(hidden_dim, 1)
        self.expand_S = nn.Linear(1, hidden_dim)

        # Decoder: input dimension depends on residual_type
        dec_in = hidden_dim * 2 if residual_type == "concat" else hidden_dim
        decoder_layers = []
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(dec_in, hidden_dim))
            decoder_layers.append(nn.ReLU())
            dec_in = hidden_dim
        decoder_layers.append(nn.Linear(dec_in, 1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Loss functions and optimizer
        self.loss_z = nn.MSELoss()  # MSE(S_pred, z)
        self.loss_y = nn.BCELoss()  # BCE(y_pred, y)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
          - Encode x through encoder layers, collect residuals.
          - Compute scalar S from final encoder output, then expand to vector.
          - Combine expanded S with last encoder latent.
          - Decode to y_hat.
        Returns:
            S (Tensor [batch,1]), y_hat (Tensor [batch,1])
        """
        residuals = []
        h = x
        for layer in self.feature_layers:
            h = layer(h)
            residuals.append(h)

        # Compute S and expanded S
        S = self.to_S(h)            # [batch,1]
        exp_S = self.expand_S(S)    # [batch,hidden_dim]

        # Combine expanded S with last encoder output
        last = residuals[-1]        # [batch,hidden_dim]
        if self.residual_type == "sum":
            combined = exp_S + last
        else:  # "concat"
            combined = torch.cat([exp_S, last], dim=1)

        y_hat = self.decoder(combined)  # [batch,1]
        return S, y_hat

    def train_model(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        x_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        z_val: torch.Tensor = None,
        early_stopping: int = 10,
        record_loss: bool = False,
    ):
        """
        Joint training with combined loss: MSE(S_pred, z) + BCE(y_pred, y).
        Supports optional validation and early stopping on BCE.

        Args:
            x (Tensor): Training features [N, D].
            y (Tensor): Main labels [N, 1].
            z (Tensor): Privileged side info [N, 1].
            x_val, y_val, z_val (Tensor, optional): Validation sets.
            early_stopping (int): Patience for early stopping on validation BCE loss.
            record_loss (bool): If True, returns (train_history, val_history).
        Returns:
            (train_hist, val_hist) if record_loss else None
        """
        train_hist, val_hist = [], []
        best_val = float("inf")
        patience = 0

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            S_pred, y_pred = self.forward(x)
            loss = self.loss_z(S_pred, z) + self.loss_y(y_pred, y)
            loss.backward()
            self.optimizer.step()

            if record_loss:
                train_hist.append(self.loss_y(y_pred, y).item())

            # Validation step
            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    _, yv_pred = self.forward(x_val)
                    val_loss = self.loss_y(yv_pred, y_val).item()
                if record_loss:
                    val_hist.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping:
                        break

        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, x: torch.Tensor, y: torch.Tensor):
        """
        Evaluate on test set: compute accuracy, AUC, BCE loss,
        and per-class precision/recall/F1.

        Returns:
            Dict with keys: "accuracy", "auc", "bce_loss", 
            "p0", "r0", "f0", "p1", "r1", "f1"
        """
        self.eval()
        with torch.no_grad():
            _, y_pred = self.forward(x)
            y_pred_flat = y_pred.view(-1)
            y_flat = y.view(-1)
            y_hat = (y_pred_flat >= 0.5).float()

            # Accuracy
            accuracy = (y_hat == y_flat).float().mean().item()

            # AUC
            try:
                auc_score = roc_auc_score(y_flat.cpu().numpy(), y_pred_flat.cpu().numpy())
            except ValueError:
                auc_score = 0.0

            bce_loss = self.loss_y(y_pred_flat, y_flat).item()
            p, r, f, _ = precision_recall_fscore_support(
                y_flat.cpu().numpy(), y_hat.cpu().numpy(), zero_division=0
            )

        return {
            "accuracy": accuracy,
            "auc": auc_score,
            "bce_loss": bce_loss,
            "p0": p[0],
            "r0": r[0],
            "f0": f[0],
            "p1": p[1],
            "r1": r[1],
            "f1": f[1],
        }


# =========================
#     UTILITY FUNCTIONS
# =========================

def plot_loss_curve(train_losses, val_losses, filename: str):
    """
    Plot and save training vs. validation BCE loss curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train BCE Loss")
    plt.plot(val_losses, label="Validation BCE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Train vs Validation BCE Loss")
    plt.legend()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def combined_run_experiments_direct(
    model_class,
    method_name: str,
    num_runs: int,
    param_grid: dict,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_z: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    cv: int = 3,
    sample_size_per_class: int = 250,
    early_stopping: int = 10
):
    """
    Combined procedure that, for each of num_runs iterations:
      1) Samples `sample_size_per_class` positives + negatives (no overlap restriction).
      2) Runs a k-fold grid search on that sampled subset (using both X, Y, Z).
      3) Retrains model on the entire sampled subset with the best hyperparameters.
      4) Evaluates on a random 125-example subset of test pool.
      5) Records both the best_params found and the test metrics.

    Args:
        model_class: A Direct-pattern model class (expects __init__(input_dim, **hparams),
                     train_model(...), evaluate(...) methods).
        method_name: String name used for saving loss curves (e.g., “Simul_NoDecomp”).
        num_runs: Number of repeated experiments.
        param_grid: Dict of hyperparameter lists to search (same format as before).
        train_x, train_y, train_z: Tensors for the entire training pool.
        test_x, test_y: Tensors for the entire test pool.
        cv: Number of folds for grid search within each run.
        sample_size_per_class: How many positives/negatives to sample each run.
        early_stopping: Patience parameter to pass into train_model.

    Returns:
        metrics_list: List of dicts (length = num_runs), each from model.evaluate().
        best_params_list: List of dicts (length = num_runs), the best_params found each run.
    """
    metrics_list = []
    best_params_list = []

    # Build a DataFrame to find indices of class 0 vs 1 in the train pool
    df_pool = pd.DataFrame(train_x.numpy())
    df_pool["Diabetes_binary"] = train_y.numpy().flatten()
    class1_indices = df_pool[df_pool["Diabetes_binary"] == 1].index.tolist()
    class0_indices = df_pool[df_pool["Diabetes_binary"] == 0].index.tolist()

    test_pool_size = len(test_x)

    for run_idx in range(1, num_runs + 1):
        # ----------------------------
        # 1) Balanced sampling for this run
        # ----------------------------
        sampled_pos = random.sample(
            class1_indices, min(len(class1_indices), sample_size_per_class)
        )
        sampled_neg = random.sample(
            class0_indices, min(len(class0_indices), sample_size_per_class)
        )
        train_idx = sampled_pos + sampled_neg

        x_sel = train_x[train_idx]      # [2*sample_size_per_class, D]
        y_sel = train_y[train_idx]      # [2*sample_size_per_class, 1]
        z_sel = train_z[train_idx]      # [2*sample_size_per_class, 1]

        # ----------------------------
        # 2) k-fold grid search on the sampled subset
        # ----------------------------
        best_auc = -float("inf")
        best_params = None
        keys = list(param_grid.keys())

        # Convert to NumPy for easy indexing inside KFold
        X_np = x_sel.numpy()
        Y_np = y_sel.numpy().flatten()
        Z_np = z_sel.numpy().flatten()

        # Pre-compute KFold on this subset
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for combo in itertools.product(*(param_grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            fold_aucs = []

            for tr_idx, val_idx in kf.split(X_np):
                # Build train/val fold
                x_tr = torch.tensor(X_np[tr_idx], dtype=torch.float32)
                y_tr = torch.tensor(Y_np[tr_idx], dtype=torch.float32).view(-1, 1)
                z_tr = torch.tensor(Z_np[tr_idx], dtype=torch.float32).view(-1, 1)

                x_va = torch.tensor(X_np[val_idx], dtype=torch.float32)
                y_va = torch.tensor(Y_np[val_idx], dtype=torch.float32).view(-1, 1)
                z_va = torch.tensor(Z_np[val_idx], dtype=torch.float32).view(-1, 1)

                # Instantiate model with these hyperparameters
                model = model_class(input_dim=X_np.shape[1], **params)

                # Train on fold
                model.train_model(
                    x_tr,
                    y_tr,
                    z_tr,
                    x_val=x_va,
                    y_val=y_va,
                    z_val=z_va,
                    early_stopping=early_stopping,
                    record_loss=False,
                )

                # Evaluate fold AUC on main task
                with torch.no_grad():
                    _, yv_pred = model.forward(x_va)
                    try:
                        fold_auc = roc_auc_score(
                            y_va.cpu().numpy(), yv_pred.detach().cpu().numpy()
                        )
                    except ValueError:
                        fold_auc = 0.0
                    fold_aucs.append(fold_auc)

            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params.copy()

        print(
            f"[{method_name}] Run {run_idx} → Best CV params: {best_params}, "
            f"Avg CV AUC: {best_auc:.4f}"
        )
        best_params_list.append(best_params)

        # ----------------------------
        # 3) Retrain on entire sampled subset with best_params
        # ----------------------------
        model = model_class(input_dim=x_sel.shape[1], **best_params)
        model.train_model(
            x_sel,
            y_sel,
            z_sel,
            early_stopping=early_stopping,
            record_loss=False,
        )

        # ----------------------------
        # 4) Evaluate on random 125‐example test subset
        # ----------------------------
        test_idx = random.sample(range(test_pool_size), min(125, test_pool_size))
        x_test_run = test_x[test_idx]
        y_test_run = test_y[test_idx]

        metrics = model.evaluate(x_test_run, y_test_run)
        print(f"[{method_name}] Run {run_idx} → Test metrics: {metrics}")
        metrics_list.append(metrics)

    return metrics_list, best_params_list


def aggregate_metrics(metrics_list: list):
    """
    Aggregate a list of metric dictionaries by computing mean ± std for each key.
    """
    agg = {}
    for key in metrics_list[0].keys():
        vals = [m[key] for m in metrics_list]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        agg[key] = f"{mean_val:.4f} ± {std_val:.4f}"
    return agg


def write_results_csv(filename: str, method: str, metrics: dict, params: str):
    """
    Append a row of results (metrics + params) to a CSV file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                method,
                metrics.get("auc", ""),
                metrics.get("accuracy", ""),
                metrics.get("p0", ""),
                metrics.get("r0", ""),
                metrics.get("f0", ""),
                metrics.get("p1", ""),
                metrics.get("r1", ""),
                metrics.get("f1", ""),
                metrics.get("bce_loss", ""),
                params,
            ]
        )


def print_aggregated_results(method: str, metrics: dict, params: str):
    """
    Print aggregated results in a readable format.
    """
    print("\n=== Final Aggregated Results ===")
    print(f"Method: {method}")
    print(f"AUC: {metrics.get('auc','')}")
    print(f"Accuracy: {metrics.get('accuracy','')}")
    print(f"Class0 → Precision: {metrics.get('p0','')}, Recall: {metrics.get('r0','')}, F1: {metrics.get('f0','')}")
    print(f"Class1 → Precision: {metrics.get('p1','')}, Recall: {metrics.get('r1','')}, F1: {metrics.get('f1','')}")
    print(f"BCE Loss: {metrics.get('bce_loss','')}")
    print(f"Best Parameters: {params}")
    print("---------------------------------------------------")


# =========================
#          MAIN
# =========================

def main():
    """
    Main entry point to run Direct‐pattern experiments:
    1) Load & preprocess data.
    2) For each model config, do combined grid search + repeated experiments.
    3) Aggregate & print.
    """
    # 1) Reproducibility
    set_seed(42)

    OUTPUT_CSV = "final_results.csv"

    # Initialize CSV header if it doesn’t exist
    dir_path = os.path.dirname(OUTPUT_CSV)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Method",
                    "AUC",
                    "Overall_Acc",
                    "Acc_Class0",
                    "Prec_Class0",
                    "Rec_Class0",
                    "F1_Class0",
                    "Acc_Class1",
                    "Prec_Class1",
                    "Rec_Class1",
                    "F1_Class1",
                    "BCE_Loss",
                    "Params",
                ]
            )

    # 2) Preprocess data
    dp = DirectDataPreprocessor(dataset_id=891, side_info_path="prompting/augmented_data_70B.csv")
    grid_x, grid_y, grid_z, train_x, train_y, train_z, test_x, test_y = dp.preprocess()

    # 3) Experiment definitions
    experiments = [
        {
            "name": "Simul_NoDecomp",
            "model": DirectPatternNoDecomp,
            "param_grid": {
                "hidden_dim": [64, 128, 256],
                "num_layers": [1, 2, 3, 4],
                "lr": [0.001, 0.01],
                "epochs": [100],              # fix joint epochs during grid search
                "residual_type": ["concat"],  # fix residual_type during grid search
            },
        },
        {
            "name": "Simul_with_Decomp",
            "model": DirectPatternResidual,
            "param_grid": {
                "hidden_dim": [64, 128, 256],
                "num_layers": [1, 2, 3, 4],
                "lr": [0.001, 0.01],
                "epochs": [100],
                "residual_type": ["concat"],
            },
        },
        {
            "name": "Decoupled_NoDecomp",
            "model": DirectDecoupledResidualNoDecomp,
            "param_grid": {
                "hidden_dim": [64, 128, 256],
                "num_layers": [1, 2, 3, 4],
                "lr": [0.001, 0.01],
                "epochs": [100],
                "residual_type": ["concat"],
            },
        },
        {
            "name": "Decoupled_with_Decomp",
            "model": DirectDecoupledResidualModel,
            "param_grid": {
                "hidden_dim": [64, 128, 256],
                "num_layers": [1, 2, 3, 4],
                "lr": [0.001, 0.01],
                "epochs": [100],
                "residual_type": ["concat"],
            },
        },
    ]

    # 4) Loop over experiment configs, calling the combined routine
    for cfg in experiments:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        name = cfg["name"]
        ModelClass = cfg["model"]
        param_grid = cfg["param_grid"]

        print(f"\n=== Combined Grid Search & Runs for {name} ===")
        metrics_list, best_params_list = combined_run_experiments_direct(
            model_class=ModelClass,
            method_name=name,
            num_runs=10,
            param_grid=param_grid,
            train_x=train_x,
            train_y=train_y,
            train_z=train_z,
            test_x=test_x,
            test_y=test_y,
            cv=3,
            sample_size_per_class=250,
            early_stopping=10,
        )

        # 5) Aggregate across runs
        aggregated = aggregate_metrics(metrics_list)
        # Print aggregated and (optionally) save to CSV:
        print_aggregated_results(name, aggregated, str(best_params_list))
        # Optionally: write to CSV
        # write_results_csv(OUTPUT_CSV, name, aggregated, str(best_params_list))

if __name__ == "__main__":
    main()
