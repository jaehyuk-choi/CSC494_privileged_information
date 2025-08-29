# import os
# import csv
# import random
# import itertools
# import inspect
# import numpy as np
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
#     Produces training pools and test set with auxiliary targets.
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

#         # 1. Multitask: Columns for main and auxiliary targets
#         self.target_cols = ['has_diabetes', 'health_1_10', 'diabetes_risk_score', 'Diabetes_binary']

#         # 2. Multiview: View2 columns (LLM-augmented privileged info)
#         self.view2_cont_cols = [
#             "predict_hba1c", "predict_cholesterol", "systolic_bp",
#             "diastolic_bp", "exercise_freq", "hi_sugar_freq"
#         ]
#         self.view2_cat_cols = ["employment_status"]

#     def preprocess(self):
#         """
#         Execute preprocessing and return tensors for:
#           - grid set (balanced) for hyperparameter tuning (not used in integrated CV)
#           - training pool with auxiliary targets
#           - test set (only main task)
        
#         Returns:
#             tuple: (train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
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

#         # 3) Remaining training data used as 'train pool'
#         train_pool = augmented_df.copy()

#         # 4) Construct a separate test set using original dataset (no augmented info)
#         original_df = self.X_original.copy()
#         original_df['Diabetes_binary'] = self.y_original
#         # Exclude rows that appear in augmented_df to avoid overlap
#         test_pool = original_df.drop(index=augmented_df.index, errors='ignore')
#         test_pool = test_pool[self.training_cols + ['Diabetes_binary']]

#         # 5) Fit scaler on continuous columns of train_pool, then transform train_pool + test_pool
#         self.scaler.fit(train_pool[continuous_cols])
#         train_pool_scaled = train_pool.copy()
#         train_pool_scaled[continuous_cols] = self.scaler.transform(train_pool[continuous_cols])
#         test_pool_scaled = test_pool.copy()
#         test_pool_scaled[continuous_cols] = self.scaler.transform(test_pool[continuous_cols])

#         # 6) Convert train_pool to PyTorch tensors (multi-task)
#         def df_to_tensors_multi(df):
#             features = df.drop(columns=self.target_cols, errors='ignore')
#             x = torch.from_numpy(features.values).float()
#             y_main = torch.from_numpy(df['Diabetes_binary'].values).float()
#             y_aux1 = torch.from_numpy(df['health_1_10'].values).float()
#             y_aux2 = torch.from_numpy(df['diabetes_risk_score'].values).float()
#             y_aux3 = torch.from_numpy(df['has_diabetes'].values).float()
#             return x, y_main, y_aux1, y_aux2, y_aux3

#         train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool = \
#             df_to_tensors_multi(train_pool_scaled)

#         # 7) Convert test set to tensors (only main task)
#         def df_to_tensors_test(df):
#             features = df[self.training_cols]
#             x = torch.from_numpy(features.values).float()
#             y = torch.from_numpy(df['Diabetes_binary'].values).float()
#             return x, y

#         test_x_pool, test_y_pool = df_to_tensors_test(test_pool_scaled)

#         return (
#             train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
#             test_x_pool, test_y_pool
#         )


# class MultiViewMultiTaskSimul(nn.Module):
#     """
#     Simultaneous multi-view + multi-task model:
#     - φ(x) and β(z) encode into hidden
#     - ψ shared head outputs 4: [main_x, aux1, aux2, aux3]
#     - ψ treats both s and s'; we use main_x, aux tasks from s, main_z from s'.
#     Loss = BCE(main_x, y) + BCE(main_z, y) + λ * (MSE(aux1, t1)+MSE(aux2,t2)+BCE(aux3,t3))
#     """
#     def __init__(self, dim_x, dim_z, hidden=64, nlayers_x=2, nlayers_z=2,
#                  lr=1e-3, epochs=100, lambda_aux=0.3, lambda_direct=0.0):
#         super().__init__()
#         self.epochs = epochs
#         self.lambda_aux = lambda_aux
#         # build φ
#         layers = []
#         in_dim = dim_x
#         for _ in range(nlayers_x):
#             layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
#             in_dim = hidden
#         self.phi = nn.Sequential(*layers)
#         # build β
#         layers = []
#         in_dim = dim_z
#         for _ in range(nlayers_z):
#             layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
#             in_dim = hidden
#         self.beta = nn.Sequential(*layers)
#         # shared head
#         self.shared = nn.Sequential(
#             nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4)
#         )
#         self.opt = optim.Adam(self.parameters(), lr=lr)
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def forward(self, x, z):
#         s = self.phi(x)
#         sp = self.beta(z)
#         out_x = self.shared(s)
#         out_z = self.shared(sp)
#         main_x = torch.sigmoid(out_x[:,0])
#         aux1 = out_x[:,1]
#         aux2 = out_x[:,2]
#         aux3 = torch.sigmoid(out_x[:,3])
#         main_z = torch.sigmoid(out_z[:,0])
#         return main_x, main_z, aux1, aux2, aux3

#     def compute_loss(self, main_x, main_z, aux1, aux2, aux3, y, t1, t2, t3):
#         loss_main = self.bce(main_x, y) + self.bce(main_z, y)
#         loss_aux = self.mse(aux1, t1) + self.mse(aux2, t2) + self.bce(aux3, t3)
#         return loss_main + self.lambda_aux * loss_aux

#     def train_model(self, X, Z, y, t1, t2, t3, record_loss=False):
#         losses = []
#         for e in range(self.epochs):
#             self.train()
#             self.opt.zero_grad()
#             m_x, m_z, a1, a2, a3 = self.forward(X, Z)
#             loss = self.compute_loss(m_x, m_z, a1, a2, a3, y, t1, t2, t3)
#             loss.backward()
#             self.opt.step()
#             if record_loss:
#                 losses.append(loss.item())
#         return losses if record_loss else None

#     def evaluate(self, X, Z, y):
#         self.eval()
#         with torch.no_grad():
#             m_x, m_z, *_ = self.forward(X, Z)
#         pred = ((m_x + m_z) / 2 >= 0.5).float()
#         p, r, f, _ = precision_recall_fscore_support(y.cpu(), pred.cpu(), zero_division=0)
#         auc = roc_auc_score(y.cpu(), ((m_x + m_z) / 2).cpu())
#         return {'auc': auc,
#                 'p0': p[0], 'r0': r[0], 'f0': f[0],
#                 'p1': p[1], 'r1': r[1], 'f1': f[1]}




import os
import random
import csv
import itertools

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import inspect

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import copy

def set_seed(seed: int):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class UnifiedDatasetPreprocessor:
    """
    Unified preprocessor for multiview, multitask, and direct patterns.
    - Uses existing augmented dataset unchanged.
    - Performs pattern-specific NA dropping (only on columns used by that pattern).
    - Prints missing-value counts per column and percent of rows dropped.
    - Applies common scaling to BMI + all continuous side-information.
    - Encodes any categorical side-information.
    - Splits train/test so test set excludes augmented rows by original UCI index.
    """
    def __init__(self,
                 pattern: str,
                 dataset_id: int = 891,
                 side_info_path: str = "augmented_data.csv"):
        """
        Initialize the preprocessor.

        Args:
            pattern (str): 'multiview', 'multitask', or 'direct'.
            dataset_id (int): UCI dataset ID.
            side_info_path (str): Path to CSV with augmented side-information. Must include 'index' matching UCI rows.
        """
        self.pattern = pattern.lower()
        if self.pattern not in {"multiview", "multitask", "direct"}:
            raise ValueError("pattern must be 'multiview', 'multitask', or 'direct'")
        self.dataset_id = dataset_id
        self.side_info_path = side_info_path

        # Load original UCI once
        raw = fetch_ucirepo(id=self.dataset_id)
        self.X_original = raw.data.features.copy()
        self.y_original = raw.data.targets.copy()

        # View1 (clinical) features
        self.view1_cols = [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
        ]
        # BMI-only scaling among view1
        self.view1_scaling_cols = ["BMI"]

        # Pattern-specific side-info and targets
        if self.pattern == "multiview":
            self.cont_side_cols = [
                "predict_hba1c", "predict_cholesterol", "systolic_bp",
                "diastolic_bp", "exercise_freq", "hi_sugar_freq"
            ]
            self.cat_side_cols = ["employment_status"]
            self.target_cols = ["Diabetes_binary"]
        elif self.pattern == "direct":
            self.cont_side_cols = ["predict_hba1c"]
            self.cat_side_cols = []
            self.target_cols = ["Diabetes_binary"]
        else:  # multitask
            self.cont_side_cols = []
            self.cat_side_cols = []
            self.target_cols = [
                "Diabetes_binary", "health_1_10", "diabetes_risk_score", "has_diabetes"
            ]

        # Scaler for BMI + cont side-info
        self.scaler_common = StandardScaler()
        # Encoders for cat side-info
        self.label_encoders = {col: LabelEncoder() for col in self.cat_side_cols}

    def _drop_na(self, df: pd.DataFrame, subset: list, name: str) -> pd.DataFrame:
        """
        Drop rows with NaNs in subset; print missing counts per column and rows dropped.
        """
        total = len(df)
        na_counts = df[subset].isna().sum()
        na_pct = na_counts / total * 100
        for col in subset:
            print(f"[{name}] {col}: {na_counts[col]} missing ({na_pct[col]:.2f}%)")
        df_clean = df.dropna(subset=subset).reset_index(drop=True)
        dropped = total - len(df_clean)
        print(f"[{name}] Dropped {dropped} rows ({dropped/total*100:.2f}%)\n")
        return df_clean

    def preprocess(self):
        """
        Execute preprocessing for the selected pattern.

        Returns:
            multiview: (X_train, y_train, z_train), (X_test, y_test)
            multitask: (X_train, y_main, y_aux1, y_aux2, y_aux3), (X_test, y_main)
            direct:    (X_train, y_train, z_train), (X_test, y_test)
        """
        # Load augmented data (must have 'index' col)
        aug = pd.read_csv(self.side_info_path)

        # Pattern-specific NA dropping
        if self.pattern == "multiview":
            subset = self.cont_side_cols + self.cat_side_cols + self.target_cols
        elif self.pattern == "direct":
            subset = self.cont_side_cols + self.target_cols
        else:  # multitask
            subset = self.target_cols
        aug = self._drop_na(aug, subset, self.pattern.title())

        # Common scaling for BMI + all continuous side-info
        all_cont = self.view1_scaling_cols + self.cont_side_cols
        if all_cont:
            self.scaler_common.fit(aug[all_cont])
            aug[all_cont] = self.scaler_common.transform(aug[all_cont])

        # Common encoding for categorical side-info
        for col in self.cat_side_cols:
            le = self.label_encoders[col]
            le.fit(aug[col].astype(str))
            aug[col] = le.transform(aug[col].astype(str))

        # Pattern-specific split and tensor conversion
        if self.pattern == "multiview":
            return self._finish_multiview(aug)
        elif self.pattern == "direct":
            return self._finish_direct(aug)
        else:
            return self._finish_multitask(aug)

    def _finish_multiview(self, aug: pd.DataFrame):
        # Training: entire augmented set
        train_df = aug
        # Test: exclude augmented indices
        used = train_df['index'].unique()
        X_test_df = self.X_original.drop(index=used, errors='ignore').reset_index(drop=True)
        y_test_df = self.y_original.drop(index=used, errors='ignore').reset_index(drop=True)

        # Tensors for training
        X_train = torch.tensor(train_df[self.view1_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train_df[self.target_cols[0]].values, dtype=torch.float32)
        z_train = torch.tensor(train_df[self.cont_side_cols + self.cat_side_cols].values, dtype=torch.float32)

        # Tensors for test 
        X_test = torch.tensor(X_test_df[self.view1_cols].values, dtype=torch.float32)
        y_test = torch.tensor(y_test_df.values, dtype=torch.float32)

        return (X_train, y_train, z_train), (X_test, y_test)

    def _finish_multitask(self, aug: pd.DataFrame):
        train_df = aug
        used = train_df['index'].unique()
        X_test_df = self.X_original.drop(index=used, errors='ignore').reset_index(drop=True)
        y_test_df = self.y_original.drop(index=used, errors='ignore').reset_index(drop=True)

        # Tensors for training (main + aux)
        X_train = torch.tensor(train_df[self.view1_cols].values, dtype=torch.float32)
        y_main = torch.tensor(train_df['Diabetes_binary'].values, dtype=torch.float32)
        y_aux1 = torch.tensor(train_df['health_1_10'].values, dtype=torch.float32)
        y_aux2 = torch.tensor(train_df['diabetes_risk_score'].values, dtype=torch.float32)
        y_aux3 = torch.tensor(train_df['has_diabetes'].values, dtype=torch.float32)

        # Tensors for test 
        X_test = torch.tensor(X_test_df[self.view1_cols].values, dtype=torch.float32)
        y_test = torch.tensor(y_test_df.values, dtype=torch.float32)

        return (X_train, y_main, y_aux1, y_aux2, y_aux3), (X_test, y_test)

    def _finish_direct(self, aug: pd.DataFrame):
        train_df = aug
        used = train_df['index'].unique()
        X_test_df = self.X_original.drop(index=used, errors='ignore').reset_index(drop=True)
        y_test_df = self.y_original.drop(index=used, errors='ignore').reset_index(drop=True)

        # Tensors for training
        X_train = torch.tensor(train_df[self.view1_cols].values, dtype=torch.float32)
        y_train = torch.tensor(train_df['Diabetes_binary'].values, dtype=torch.float32)
        z_train = torch.tensor(train_df[self.cont_side_cols].values, dtype=torch.float32).view(-1, len(self.cont_side_cols))

        # Tensors for test
        X_test = torch.tensor(X_test_df[self.view1_cols].values, dtype=torch.float32)
        y_test = torch.tensor(y_test_df.values, dtype=torch.float32)

        return (X_train, y_train, z_train), (X_test, y_test)

class PreFineTuneMultiView(nn.Module):
    """
    Two‐branch pre-finetune Multi‐View model:
      - φ: main branch on X (primary view).
      - β: auxiliary branch on Z (privileged view).
      - ψ: shared classifier head.

    Workflow:
      1) pretrain_beta on (Z, y) → learn β + ψ for pretrain_epochs.
      2) Freeze β, copy ψ as teacher.
      3) finetune_main on (X, y) with auxiliary regularization for finetune_epochs:
           L = BCE(ψ(φ(X)), y) + λ * BCE( ψ_teacher(β(Z)), y ).
      4) At test time, only φ + ψ are used (no Z).
    """
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        hidden: int = 32,
        num_layers_x: int = 1,
        num_layers_z: int = 1,
        lr: float = 1e-3,
        pretrain_epochs: int = 100,
        finetune_epochs: int = 100,
        lambda_aux: float = 0.5
    ):
        """
        Args:
            dim_x (int): Dimensionality of X.
            dim_z (int): Dimensionality of Z.
            hidden (int): Hidden size for both φ and β.
            num_layers_x (int): Number of layers in φ.
            num_layers_z (int): Number of layers in β.
            lr (float): Learning rate for both stages.
            pretrain_epochs (int): Epochs for β+ψ pretraining.
            finetune_epochs (int): Epochs for φ+ψ finetuning.
            lambda_aux (float): Weight for auxiliary loss term.
        """
        super().__init__()
        self.pretrain_epochs  = pretrain_epochs
        self.finetune_epochs  = finetune_epochs
        self.lambda_aux       = lambda_aux

        # φ network: X → hidden
        self.phi = self._make_mlp(dim_x, hidden, num_layers_x)
        # β network: Z → hidden
        self.beta = self._make_mlp(dim_z, hidden, num_layers_z)

        # ψ classifier: hidden → [0,1]
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

        # Loss & optimizers
        self.bce = nn.BCELoss()
        # Pretraining optimizer for β + ψ
        self.pretrain_opt = optim.Adam(
            list(self.beta.parameters()) + list(self.classifier.parameters()),
            lr=lr
        )
        # Placeholder; will be set after pretrain_beta()
        self.finetune_opt = None

        # Placeholder for teacher ψ after β is frozen
        self.teacher = None

    def _make_mlp(self, in_dim, out_dim, n_layers):
        layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(out_dim, out_dim), nn.ReLU()]
        return nn.Sequential(*layers)

    def forward_beta(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Pass Z through β → ψ to get auxiliary predictions.
        """
        h_z = self.beta(Z)
        return self.classifier(h_z).view(-1)

    def forward_main(self, X: torch.Tensor) -> torch.Tensor:
        """
        Pass X through φ → ψ to get main predictions.
        """
        h_x = self.phi(X)
        return self.classifier(h_x).view(-1)

    def pretrain_beta(self, Z: torch.Tensor, y: torch.Tensor):
        """
        Pretrain β + ψ on (Z, y) for self.pretrain_epochs,
        then freeze β and copy ψ as the teacher.
        """
        for _ in range(self.pretrain_epochs):
            self.pretrain_opt.zero_grad()
            y_hat_z = self.forward_beta(Z)
            loss = self.bce(y_hat_z, y)
            loss.backward()
            self.pretrain_opt.step()

        # Freeze β parameters
        for p in self.beta.parameters():
            p.requires_grad = False

        # Copy ψ to teacher for auxiliary signals
        self.teacher = copy.deepcopy(self.classifier)

        # Initialize finetune optimizer for φ + ψ
        lr = self.pretrain_opt.defaults["lr"]
        self.finetune_opt = optim.Adam(
            list(self.phi.parameters()) + list(self.classifier.parameters()),
            lr=lr
        )

    def finetune_main(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        y: torch.Tensor,
        x_val: torch.Tensor = None,
        z_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        early_stopping_patience: int = 10
    ):
        """
        Finetune φ + ψ on (X, y) for self.finetune_epochs,
        using auxiliary regularization from teacher(β(Z)).
        Optionally apply early stopping on validation φ-branch loss.
        """
        best_val_loss = float('inf')
        patience = 0

        for _ in range(self.finetune_epochs):
            self.train()
            self.finetune_opt.zero_grad()

            # Main prediction branch
            y_hat_x = self.forward_main(X)
            # Teacher predictions from frozen β + teacher ψ
            with torch.no_grad():
                y_teacher = self.teacher(self.beta(Z)).view(-1)

            # Compute combined loss
            loss_main = self.bce(y_hat_x, y)
            loss_aux  = self.bce(y_hat_x, y_teacher)
            loss = loss_main + self.lambda_aux * loss_aux

            loss.backward()
            self.finetune_opt.step()

            # Early stopping check on φ-branch validation loss
            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    y_val_hat = self.forward_main(x_val)
                    val_loss = self.bce(y_val_hat, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break  # stop finetuning early

    def evaluate_main(self, X: torch.Tensor, y: torch.Tensor):
        """
        Compute AUC and per-class precision/recall/F1 for φ-branch.
        Returns metrics for both class 0 and class 1.
        """
        self.eval()
        with torch.no_grad():
            y_score = self.forward_main(X).cpu().numpy()
        y_true = y.cpu().numpy()
        print(y_score)
        y_pred = (y_score >= 0.5).astype(int)

        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
        }

# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------
def run_experiments_multiview(
    model_class,
    method_name: str,
    num_runs: int,
    param_grid: dict,
    train_X_pool, train_Z_pool, train_y_pool,
    test_X_pool, test_y_pool,
    cv: int = 3
):
    """
    1) Sample 250 pos + 250 neg from train_pool (no overlap across runs).
    2) Sample 125 test points from test_X_pool/test_y_pool.
    3) Grid‐search with cv folds on (X,Z,y) to pick best hyperparams.
    4) Retrain on all 500 with best hyperparams.
    5) Evaluate φ-branch on test 125 → collect metrics.
    6) Repeat num_runs times.
    """
    metrics_list = []
    best_params_list = []

    pos_idxs = (train_y_pool==1).nonzero(as_tuple=True)[0].tolist()
    neg_idxs = (train_y_pool==0).nonzero(as_tuple=True)[0].tolist()
    used = set()
    N_test = len(test_X_pool)

    for run in range(1, num_runs+1):
        # 1) sample train
        avail_p = [i for i in pos_idxs if i not in used]
        avail_n = [i for i in neg_idxs if i not in used]
        samp_p = random.sample(avail_p, 250)
        samp_n = random.sample(avail_n, 250)
        idx500 = samp_p + samp_n
        used.update(idx500)
        X500 = train_X_pool[idx500]
        Z500 = train_Z_pool[idx500]
        y500 = train_y_pool[idx500]

        # 2) sample test
        idx125 = random.sample(range(N_test), 125)
        X125 = test_X_pool[idx125]
        y125 = test_y_pool[idx125]

        # 3) grid‐search
        best_auc = -np.inf
        best_params = None
        keys = list(param_grid.keys())
        for combo in itertools.product(*(param_grid[k] for k in keys)):
            params = dict(zip(keys, combo))
            aucs = []
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            for tr, va in kf.split(X500):
                X_tr, Z_tr, y_tr = X500[tr], Z500[tr], y500[tr]
                X_va, Z_va, y_va = X500[va], Z500[va], y500[va]

                m = model_class(dim_x=X_tr.shape[1],
                                dim_z=Z_tr.shape[1],
                                **params)
                m.pretrain_beta(Z_tr, y_tr)
                m.finetune_main(X_tr, Z_tr, y_tr,
                                x_val=X_va, z_val=Z_va, y_val=y_va)
                metrics = m.evaluate_main(X_va, y_va)
                aucs.append(metrics["auc"])

            mean_auc = float(np.mean(aucs))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params

        print(f"[{method_name}] Run {run}: best_params={best_params}, CV AUC={best_auc:.4f}")
        best_params_list.append(best_params)

        # 4) retrain final
        final = model_class(dim_x=X500.shape[1],
                            dim_z=Z500.shape[1],
                            **best_params)
        final.pretrain_beta(Z500, y500)
        final.finetune_main(X500, Z500, y500)

        # 5) test evaluation
        test_metrics = final.evaluate_main(X125, y125)
        print(f"[{method_name}] Run {run}: test_metrics={test_metrics}")
        metrics_list.append(test_metrics)

    return metrics_list, best_params_list

# -----------------------------------------------------------------------------
# Metrics aggregation & I/O
# -----------------------------------------------------------------------------
def aggregate_metrics(metrics_list):
    """
    Compute mean ± std for each metric key.
    """
    agg = {}
    for k in metrics_list[0]:
        vals = [m[k] for m in metrics_list]
        agg[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    return agg

def write_results_csv(fname, method, metrics, params):
    with open(fname, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            method,
            metrics.get('auc',''),
            metrics.get('p0',''), metrics.get('r0',''), metrics.get('f0',''),
            metrics.get('p1',''), metrics.get('r1',''), metrics.get('f1',''),
            params
        ])

def print_aggregated_results(method, metrics, params):
    print("\n=== Final Aggregated Results ===")
    print(f"Method: {method}")
    print(f"AUC: {metrics.get('auc','')}")
    print(f"Class0 → P: {metrics.get('p0','')}, R: {metrics.get('r0','')}, F1: {metrics.get('f0','')}")
    print(f"Class1 → P: {metrics.get('p1','')}, R: {metrics.get('r1','')}, F1: {metrics.get('f1','')}")
    print(f"Best params per run: {params}")
    print("-"*50)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    set_seed(42)

    OUTPUT_CSV = "multiview_results.csv"
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    # header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Method","AUC","p0","r0","f0","p1","r1","f1","Params"])

    # preprocess
    dp = UnifiedDatasetPreprocessor(
        pattern="multiview",
        dataset_id=891,
        side_info_path="prompting/augmented_data.csv"
    )
    (X_tr, y_tr, Z_tr), (X_te, y_te) = dp.preprocess()

    # exp config
    experiments = [
        {
            "name": "PreFineTune",
            "model": PreFineTuneMultiView,
            "param_grid": {
                "hidden": [128],
                "num_layers_x": [1],
                "num_layers_z": [1],
                "lr": [0.001],
                "pretrain_epochs": [100],
                "finetune_epochs": [100],
                "lambda_aux": [0.3]
            }
        }
    ]

    for cfg in experiments:
        name = cfg["name"]
        Model = cfg["model"]
        grid = cfg["param_grid"]
        print(f"\n=== Running {name} ===")
        metrics_list, best_params_list = run_experiments_multiview(
            model_class=Model,
            method_name=name,
            num_runs=10,
            param_grid=grid,
            train_X_pool=X_tr,
            train_Z_pool=Z_tr,
            train_y_pool=y_tr,
            test_X_pool=X_te,
            test_y_pool=y_te,
            cv=3
        )
        agg = aggregate_metrics(metrics_list)
        write_results_csv(OUTPUT_CSV, name, agg, best_params_list)
        print_aggregated_results(name, agg, best_params_list)

if __name__ == "__main__":
    main()