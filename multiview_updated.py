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


# =============================================================================
#                               UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int):
    """
    Fix random seed for reproducibility across random, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_loss_curve(train_losses: list, val_losses: list, filename: str):
    """
    Plot and save training vs. validation loss curve to a file.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()

    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    plt.savefig(filename)
    plt.close()


def grid_search_multiview(
    model_class,
    param_grid: dict,
    X: torch.Tensor,
    Z: torch.Tensor,
    y: torch.Tensor,
    cv: int = 3
):
    """
    Perform grid search for multi-view models, using ROC AUC on the main task.

    Args:
        model_class: class implementing __init__(dim_x, dim_z, **params) and train_model()/evaluate().
        param_grid: dict of hyperparameter names to lists of candidate values.
        X: (N, dim_x) torch.Tensor of view1 features.
        Z: (N, dim_z) torch.Tensor of view2 features.
        y: (N,) torch.Tensor of binary targets.
        cv: number of cross-validation folds.

    Returns:
        best_params: dict of hyperparameters achieving highest average AUC.
        best_auc: float, corresponding AUC.
    """
    best_auc = -np.inf
    best_params = None
    keys = list(param_grid.keys())

    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        aucs = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            x_tr, z_tr, y_tr = X[train_idx], Z[train_idx], y[train_idx]
            x_va, z_va, y_va = X[val_idx], Z[val_idx], y[val_idx]

            # Instantiate model with dynamic signature
            sig = inspect.signature(model_class.__init__).parameters
            init_kwargs = {}
            if "dim_x" in sig:
                init_kwargs["dim_x"] = X.shape[1]
            if "dim_z" in sig:
                init_kwargs["dim_z"] = Z.shape[1]
            for k, v in params.items():
                if k in sig:
                    init_kwargs[k] = v

            model = model_class(**init_kwargs)

            # If auxiliary pretraining method exists, call it
            if hasattr(model, "pretrain_aux"):
                model.pretrain_aux(z_tr, y_tr)

            # Train main branch (no loss recording)
            model.train_model(x_tr, z_tr, y_tr, record_loss=False)

            # Validate to compute AUC
            model.eval()
            with torch.no_grad():
                out = model.forward(x_va, z_va)[0]  # first output is prediction from view1 branch
            try:
                auc_score = roc_auc_score(y_va.cpu().numpy(), out.cpu().numpy())
            except:
                auc_score = 0.0
            aucs.append(auc_score)

        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params.copy()

    return best_params, best_auc


def run_experiments_multiview(
    model_class,
    method_name: str,
    num_runs: int,
    best_params: dict,
    train_X: torch.Tensor,
    train_Z: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    test_Z: torch.Tensor,
    test_y: torch.Tensor
):
    """
    Run multiple repeated experiments for a multi-view model under best hyperparameters.

    For each run:
      1) Balanced sampling of up to 250 examples per class from train pool.
      2) 80/20 train/validation split on that subset.
      3) Auxiliary pretraining if applicable, then train with loss recording.
      4) Plot and save loss curves.
      5) Evaluate on a random 125-example subset of test pool.

    Args:
        model_class: class implementing the train/evaluate interface.
        method_name: string to identify model (used in loss filenames).
        num_runs: number of repeated experiments.
        best_params: dict of hyperparameters obtained from grid search.
        train_X, train_Z, train_y: (N_train, dim_x), (N_train, dim_z), (N_train,) training pool.
        test_X, test_Z, test_y: (N_test, dim_x), (N_test, dim_z), (N_test,) test pool.

    Returns:
        metrics_list: list of dicts of evaluation metrics for each run.
    """
    metrics_list = []

    # Identify positive and negative indices in the training pool
    y_np = train_y.cpu().numpy()
    idx_pos = np.where(y_np == 1)[0].tolist()
    idx_neg = np.where(y_np == 0)[0].tolist()
    test_pool_size = len(test_X)

    for run in range(1, num_runs + 1):
        # 1) Balanced sampling (max 250 each)
        sampled1 = random.sample(idx_pos, min(len(idx_pos), 250))
        sampled0 = random.sample(idx_neg, min(len(idx_neg), 250))
        idx_train = sampled1 + sampled0

        x_pool = train_X[idx_train]
        z_pool = train_Z[idx_train]
        y_pool = train_y[idx_train]

        # 2) Random 125-example subset from test pool
        test_idx = random.sample(range(test_pool_size), min(125, test_pool_size))
        x_test_run = test_X[test_idx]
        y_test_run = test_y[test_idx]
        z_test_run = test_Z[test_idx]

        # 3) 80/20 train/validation split
        perm = torch.randperm(len(x_pool))
        split = int(0.8 * len(x_pool))
        tr_idx, va_idx = perm[:split], perm[split:]
        x_tr, z_tr, y_tr = x_pool[tr_idx], z_pool[tr_idx], y_pool[tr_idx]
        x_va, z_va, y_va = x_pool[va_idx], z_pool[va_idx], y_pool[va_idx]

        # 4) Instantiate model dynamically
        sig = inspect.signature(model_class.__init__).parameters
        init_kwargs = {}
        if "dim_x" in sig:
            init_kwargs["dim_x"] = train_X.shape[1]
        if "dim_z" in sig:
            init_kwargs["dim_z"] = train_Z.shape[1]
        for k, v in best_params.items():
            if k in sig:
                init_kwargs[k] = v

        model = model_class(**init_kwargs)

        # 5) Auxiliary pretraining if available
        if hasattr(model, "pretrain_aux"):
            model.pretrain_aux(z_tr, y_tr)

        # 6) Train with loss recording
        train_val = model.train_model(
            x_tr, z_tr, y_tr,
            x_val=x_va, y_val=y_va, z_val=z_va,
            record_loss=True
        )
        train_losses, val_losses = train_val

        # 7) Save loss curve figure
        filename = f"img/{method_name}_run{run}.png"
        plot_loss_curve(train_losses, val_losses, filename)

        # 8) Evaluate on test subset
        metric = model.evaluate(x_test_run, y_test_run, z_test_run)
        metrics_list.append(metric)
        print(f"[{method_name}] Run {run} → {metric}")

    return metrics_list


def aggregate_metrics(metrics_list: list):
    """
    Aggregate a list of metric dictionaries by computing mean ± std for each key.

    Args:
        metrics_list: list of dicts, all sharing the same keys.

    Returns:
        agg: dict mapping each key to a string "mean ± std".
    """
    agg = {}
    for key in metrics_list[0].keys():
        vals = [m[key] for m in metrics_list]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        agg[key] = f"{mean_val:.4f} ± {std_val:.4f}"
    return agg


def print_aggregated_results(method: str, metrics: dict, params: str):
    """
    Print the final aggregated results for a given method.

    Args:
        method: model name.
        metrics: aggregated metrics dict (mean ± std).
        params: string representation of hyperparameters.
    """
    print("\n=== Final Aggregated Results ===")
    print(f"Method: {method}")
    if "accuracy" in metrics:
        print(f"Overall Accuracy: {metrics.get('accuracy', '')}")
    if "accuracy_class0" in metrics:
        print(f"Class0 Accuracy: {metrics.get('accuracy_class0', '')}")
    if "accuracy_class1" in metrics:
        print(f"Class1 Accuracy: {metrics.get('accuracy_class1', '')}")
    print(f"AUC: {metrics.get('auc', '')}")
    print(f"Class0 → Precision: {metrics.get('p0', '')}, Recall: {metrics.get('r0', '')}, F1: {metrics.get('f0', '')}")
    print(f"Class1 → Precision: {metrics.get('p1', '')}, Recall: {metrics.get('r1', '')}, F1: {metrics.get('f1', '')}")
    print(f"Best Params: {params}")
    print("---------------------------------------------------")


# =============================================================================
#                         DATA PREPROCESSOR CLASS
# =============================================================================

class MultiViewDatasetPreprocessor:
    """
    Preprocessor for multi-view learning without one-hot encoding for view2 categorical features.

    - During training: use augmented CSV (side info) for grid set & train pool.
    - Test set is drawn from original UCI data; test's view2 is a zero tensor.
    - Scale only 'BMI' among view1 features; leave other view1 features unchanged.
    - Scale continuous view2 features and label-encode categorical view2 features.
    """

    def __init__(self, dataset_id: int = 891, side_info_path: str = "prompting/augmented_data.csv"):
        """
        Args:
            dataset_id: UCI dataset ID.
            side_info_path: path to LLM-augmented CSV.
        """
        # Fetch original UCI dataset
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features  # pandas DataFrame
        self.y_original = self.original_data.data.targets   # pandas Series

        # Side-information CSV path
        self.side_info_path = side_info_path

        # View1 columns (original clinical features)
        self.view1_cols = [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
        ]
        self.view1_scaling_cols = ["BMI"]

        # View2 columns (LLM-augmented privileged info)
        self.view2_cont_cols = [
            "predict_hba1c", "predict_cholesterol", "systolic_bp",
            "diastolic_bp", "exercise_freq", "hi_sugar_freq"
        ]
        self.view2_cat_cols = ["employment_status"]

        # Main target column
        self.target = "Diabetes_binary"

        # Scalers and label encoders
        self.scaler_bmi = StandardScaler()
        self.scaler_view2_cont = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in self.view2_cat_cols}

    def preprocess(self):
        """
        Execute preprocessing and return:
          - grid set: (grid_x, grid_y, grid_z)
          - train pool: (train_x, train_y, train_z)
          - test set: (test_x, test_y, test_z)
        """
        # 1) Load augmented CSV that includes view2 and target
        augmented_df = pd.read_csv(self.side_info_path)

        # 2) Identify missing values in view2 and target, print stats
        cols_to_check = self.view2_cont_cols + self.view2_cat_cols + [self.target]
        na_counts = augmented_df[cols_to_check].isna().sum()
        na_ratio = (na_counts / len(augmented_df)) * 100
        col_dtypes = augmented_df[cols_to_check].dtypes
        for col in cols_to_check:
            print(f"{col}: {na_counts[col]} missing ({na_ratio[col]:.2f}%) — dtype: {col_dtypes[col]}")

        # 3) Fill missing values in continuous and categorical view2 and target with median
        for col in cols_to_check:
            if augmented_df[col].isna().any():
                median_value = augmented_df[col].median()
                augmented_df[col].fillna(median_value, inplace=True)

        # 4) Create balanced 'grid' set (max 375 samples per class)
        pos_idx = augmented_df[augmented_df[self.target] == 1].index.tolist()
        neg_idx = augmented_df[augmented_df[self.target] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
        grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
        grid_df = pd.concat([grid_pos, grid_neg], ignore_index=True)

        # Remaining rows form the train pool
        train_pool = augmented_df.drop(index=grid_df.index, errors="ignore").reset_index(drop=True)

        # 5) Construct test set from original UCI data (no privileged view2)
        original_df = self.X_original.copy()
        original_df[self.target] = self.y_original.values
        test_pool_df = original_df.drop(index=augmented_df.index, errors="ignore").reset_index(drop=True)

        # 6) Scale view1's BMI
        self.scaler_bmi.fit(train_pool[["BMI"]])

        def process_view1(df: pd.DataFrame) -> pd.DataFrame:
            df_copy = df.copy()
            df_copy["BMI"] = self.scaler_bmi.transform(df[["BMI"]])
            return df_copy

        grid_df_proc = process_view1(grid_df)
        train_pool_df_proc = process_view1(train_pool)
        test_pool_df_proc = process_view1(test_pool_df)

        # 7) Scale continuous view2 features using train pool
        self.scaler_view2_cont.fit(train_pool[self.view2_cont_cols])
        grid_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(grid_df[self.view2_cont_cols])
        train_pool[self.view2_cont_cols] = self.scaler_view2_cont.transform(train_pool[self.view2_cont_cols])

        # 8) Label-encode categorical view2 features
        for col in self.view2_cat_cols:
            self.label_encoders[col].fit(augmented_df[col].astype(str))
            grid_df[col] = self.label_encoders[col].transform(grid_df[col].astype(str))
            train_pool[col] = self.label_encoders[col].transform(train_pool[col].astype(str))

        # 9) Combine view2 continuous + categorical into a DataFrame, then to Tensor
        grid_view2 = grid_df[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)
        train_view2 = train_pool[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)

        grid_z = torch.tensor(grid_view2.values, dtype=torch.float32)
        train_z = torch.tensor(train_view2.values, dtype=torch.float32)
        if grid_z.ndimension() == 1:
            grid_z = grid_z.unsqueeze(1)
            train_z = train_z.unsqueeze(1)

        # 10) Convert view1 and target for grid and train to Tensors
        def to_tensor(df: pd.DataFrame, feature_cols: list, target_col: str = None):
            X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            y = torch.tensor(df[target_col].values, dtype=torch.float32).view(-1) if target_col else None
            return X, y

        grid_x, grid_y = to_tensor(grid_df_proc, self.view1_cols, self.target)
        train_x, train_y = to_tensor(train_pool_df_proc, self.view1_cols, self.target)
        test_x, test_y = to_tensor(test_pool_df_proc, self.view1_cols, self.target)

        # 11) Test set's view2 is a zero tensor (no privileged info)
        dim_view2 = grid_z.shape[1] if grid_z.ndimension() > 1 else 0
        test_z = torch.zeros((test_x.shape[0], dim_view2), dtype=torch.float32)

        return (grid_x, grid_y, grid_z), (train_x, train_y, train_z), (test_x, test_y, test_z)

################################################################################
# 1) Two‐Loss Multi‐View Model
################################################################################
class MultiViewNN_TwoLoss(nn.Module):
    """
    Two-branch multi-view model with independent depths:
      - β: auxiliary branch on Z (privileged view)
      - φ: main branch on X (primary view)
      - ψ: shared classifier head

    Training steps:
      1) Pretrain β + ψ on (Z, y) to predict y from side information.
      2) Freeze β and copy ψ to a fixed “teacher” classifier.
      3) Train φ + ψ on (X, y) with auxiliary regularization:
           L = BCE(ψ(φ(x)), y) + λ * BCE( teacher(β(z)), y )
    """

    def __init__(self, dim_x: int, dim_z: int, hidden: int = 32,
                 num_layers_x: int = 2, num_layers_z: int = 1,
                 lr: float = 1e-3, epochs: int = 100, lambda_aux: float = 0.5):
        """
        Args:
            dim_x (int): Dimension of view1 features (X).
            dim_z (int): Dimension of view2 features (Z).
            hidden (int): Hidden dimension for both branches.
            num_layers_x (int): Number of hidden layers in φ (main branch).
            num_layers_z (int): Number of hidden layers in β (aux branch).
            lr (float): Learning rate for optimizer.
            epochs (int): Number of epochs for both pretraining and main training.
            lambda_aux (float): Weight for auxiliary BCE loss term.
        """
        super().__init__()
        self.epochs = epochs
        self.lambda_aux = lambda_aux

        # Build φ network (main branch on X)
        self.phi = self._make_mlp(in_dim=dim_x, out_dim=hidden, n_layers=num_layers_x)

        # Build β network (auxiliary branch on Z)
        self.beta = self._make_mlp(in_dim=dim_z, out_dim=hidden, n_layers=num_layers_z)

        # Shared classifier ψ (maps hidden → scalar probability)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

        # Optimizers: 
        #   - aux_opt for pretraining β + ψ
        #   - main_opt for training φ + ψ after pretraining
        self.aux_opt = optim.Adam(
            list(self.beta.parameters()) + list(self.classifier.parameters()),
            lr=lr
        )
        self.main_opt = None

        # Loss function
        self.bce = nn.BCELoss()

        # After pretraining, this holds a frozen copy of ψ
        self.fixed_classifier = None

    def _make_mlp(self, in_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
        """
        Helper to build an MLP of n_layers layers, each mapping → out_dim then ReLU.
        """
        layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(out_dim, out_dim), nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Forward pass:
          - Compute φ(x) → hidden_x
          - Compute β(z) → hidden_z
          - Compute ψ(hidden_x) → y_x
          - Compute ψ(hidden_z) → y_z
        Returns:
            y_x (Tensor [batch]), y_z (Tensor [batch])
        """
        hidden_x = self.phi(x)
        hidden_z = self.beta(z)

        y_x = self.classifier(hidden_x).view(-1)  # predictions from X branch
        y_z = self.classifier(hidden_z).view(-1)  # predictions from Z branch
        return y_x, y_z

    def pretrain_aux(self, Z: torch.Tensor, y: torch.Tensor):
        """
        Pretrain auxiliary branch β + classifier ψ on (Z, y).
        Freezes β afterward and creates a fixed copy of ψ to act as “teacher”.
        """
        # Create dummy input for φ (not used during aux pretraining)
        dummy_x = torch.zeros(Z.size(0), self.phi[0].in_features, device=Z.device)

        for _ in range(self.epochs):
            self.aux_opt.zero_grad()
            _, y_pred_z = self.forward(dummy_x, Z)
            loss = self.bce(y_pred_z, y.view(-1))
            loss.backward()
            self.aux_opt.step()

        # Freeze all parameters in β (aux branch)
        for param in self.beta.parameters():
            param.requires_grad = False

        # Make a deep copy of classifier ψ as “teacher”
        self.fixed_classifier = copy.deepcopy(self.classifier)

        # Initialize main optimizer for φ + ψ
        self.main_opt = optim.Adam(
            list(self.phi.parameters()) + list(self.classifier.parameters()),
            lr=self.aux_opt.defaults['lr']
        )

    def train_model(self,
                    X: torch.Tensor,
                    Z: torch.Tensor,
                    y: torch.Tensor,
                    x_val: torch.Tensor = None,
                    z_val: torch.Tensor = None,
                    y_val: torch.Tensor = None,
                    record_loss: bool = False,
                    early_stopping_patience: int = 10):
        """
        Train φ + ψ on X with auxiliary regularization from fixed β + ψ.

        Args:
            X (Tensor): View1 training features.
            Z (Tensor): View2 training features.
            y (Tensor): Labels for training.
            x_val, z_val, y_val (Tensor): Optional validation sets for early stopping.
            record_loss (bool): If True, returns (train_losses, val_losses).
            early_stopping_patience (int): Patience for early stopping on validation BCE loss.

        Returns:
            (train_losses, val_losses) if record_loss else None
        """
        if self.fixed_classifier is None:
            raise RuntimeError("Must call pretrain_aux() before training main branch.")

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for _ in range(self.epochs):
            self.train()
            self.main_opt.zero_grad()

            # Forward pass on training data
            y_pred_x, _ = self.forward(X, Z)
            with torch.no_grad():
                # Teacher predictions from fixed β + ψ
                y_teacher = self.fixed_classifier(self.beta(Z)).view(-1)

            # Compute combined loss
            loss_main = self.bce(y_pred_x, y.view(-1))
            loss_aux = self.bce(y_pred_x, y_teacher)
            loss = loss_main + self.lambda_aux * loss_aux

            loss.backward()
            self.main_opt.step()

            if record_loss:
                train_losses.append(loss.item())

            # Early stopping on validation set
            if x_val is not None and y_val is not None and z_val is not None:
                self.eval()
                with torch.no_grad():
                    y_val_pred, _ = self.forward(x_val, z_val)
                    val_loss = self.bce(y_val_pred, y_val.view(-1))
                if record_loss:
                    val_losses.append(val_loss.item())

                # Check improvement
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break

        if record_loss:
            return train_losses, val_losses

    def evaluate(self, X: torch.Tensor, y: torch.Tensor, Z: torch.Tensor):
        """
        Evaluate model on test set: compute AUC and per-class precision/recall/F1.

        Args:
            X (Tensor): View1 test features.
            y (Tensor): Test labels.
            Z (Tensor): View2 test features (zeros for test).

        Returns:
            Dict with keys: "auc", "p0", "r0", "f0", "p1", "r1", "f1"
        """
        self.eval()
        with torch.no_grad():
            y_pred, _ = self.forward(X, Z)
        y_true = y.view(-1).cpu().numpy()
        y_score = y_pred.cpu().numpy()
        y_hat = (y_score >= 0.5).astype(float)

        # Compute per-class metrics
        p, r, f, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
        }


################################################################################
# 2) Simultaneous Two‐Loss Multi‐View Model
################################################################################
class MultiViewNN_Simul(nn.Module):
    """
    Two‐branch simultaneous multi‐view model:
      L = BCE(ψ(φ(x)), y) + λ * BCE(ψ(β(z)), y)

    Both φ (view1 on X) and β (view2 on Z) are MLPs of configurable depth.
    """
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        hidden: int = 32,
        num_layers_x: int = 2,
        num_layers_z: int = 1,
        lr: float = 1e-3,
        epochs: int = 100,
        lambda_aux: float = 0.5
    ):
        """
        Args:
            dim_x (int): Input dimensionality of view1 (X).
            dim_z (int): Input dimensionality of view2 (Z).
            hidden (int): Hidden‐unit size for all intermediate layers.
            num_layers_x (int): Number of hidden layers in φ (view1 branch).
            num_layers_z (int): Number of hidden layers in β (view2 branch).
            lr (float): Learning rate for Adam optimizer.
            epochs (int): Total epochs to train.
            lambda_aux (float): Weight of the auxiliary BCE loss from view2.
        """
        super().__init__()
        self.epochs     = epochs
        self.lambda_aux = lambda_aux

        # Build φ (view1 → hidden) as an MLP with num_layers_x hidden layers
        # First layer maps dim_x → hidden, then (num_layers_x−1) layers hidden→hidden
        phi_layers = []
        in_dim = dim_x
        for i in range(num_layers_x):
            phi_layers.append(nn.Linear(in_dim, hidden))
            phi_layers.append(nn.ReLU())
            in_dim = hidden
        self.phi = nn.Sequential(*phi_layers)

        # Build β (view2 → hidden) as an MLP with num_layers_z hidden layers
        in_dim = dim_z
        beta_layers = []
        for i in range(num_layers_z):
            beta_layers.append(nn.Linear(in_dim, hidden))
            beta_layers.append(nn.ReLU())
            in_dim = hidden
        self.beta = nn.Sequential(*beta_layers)

        # Shared classifier ψ: hidden → 1 (sigmoid)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

        # Single optimizer for all parameters (φ, β, ψ)
        self.opt = optim.Adam(self.parameters(), lr=lr)

        # BCE loss
        self.bce = nn.BCELoss()

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Forward pass:
          - φ(x) → hidden_x
          - β(z) → hidden_z
          - ψ(hidden_x) = ŷ_x
          - ψ(hidden_z) = ŷ_z
        Returns:
            (ŷ_x_flat, ŷ_z_flat) both of shape [batch]
        """
        # Compute hidden representation from view1
        hidden_x = self.phi(x)      # [batch, hidden]
        # Compute hidden representation from view2
        hidden_z = self.beta(z)     # [batch, hidden]

        # Classifier produces scalar probabilities
        y_x = self.classifier(hidden_x).view(-1)  # [batch]
        y_z = self.classifier(hidden_z).view(-1)  # [batch]
        return y_x, y_z

    def train_model(
        self,
        X: torch.Tensor,
        Z: torch.Tensor,
        y: torch.Tensor,
        x_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        z_val: torch.Tensor = None,
        early_stopping_patience: int = 10,
        record_loss: bool = False
    ):
        """
        Train both branches (φ + β + ψ) jointly. Each step:
          L = BCE(ψ(φ(X)), y) + λ_aux * BCE(ψ(β(Z)), y)

        Args:
            X (Tensor): [N, dim_x] training features for view1.
            Z (Tensor): [N, dim_z] training features for view2.
            y (Tensor): [N] binary labels (0/1).
            x_val, z_val, y_val: optional validation splits.
            early_stopping_patience (int): patience on validation BCE of φ‐branch.
            record_loss (bool): if True, return (train_loss_history, val_loss_history).
        Returns:
            If record_loss: (train_loss_history: List[float], val_loss_history: List[float])
            Else: None
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 0

        for epoch in range(self.epochs):
            self.train()
            self.opt.zero_grad()

            # Forward pass on training batch
            y_pred_x, y_pred_z = self.forward(X, Z)  # each [N]

            # Compute combined loss
            loss_x = self.bce(y_pred_x, y)
            loss_z = self.bce(y_pred_z, y)
            loss = loss_x + self.lambda_aux * loss_z

            # Backprop + optimizer step
            loss.backward()
            self.opt.step()

            if record_loss:
                train_losses.append(loss.item())

            # Validation step for early stopping (only φ‐branch on X)
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    y_val_pred_x, _ = self.forward(x_val, z_val)
                    val_loss = self.bce(y_val_pred_x, y_val)
                if record_loss:
                    val_losses.append(val_loss.item())

                # Early stopping check
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break

        if record_loss:
            return train_losses, val_losses

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        Z: torch.Tensor
    ):
        """
        Evaluate on a held‐out set:
          - use only the φ‐branch output ψ(φ(X)) for computing AUC & p/r/f.

        Args:
            X (Tensor): [N_test, dim_x]
            y (Tensor): [N_test]
            Z (Tensor): [N_test, dim_z] (unused for final φ‐output but must supply)
        Returns:
            Dict with keys: "auc", "p0","r0","f0","p1","r1","f1"
        """
        self.eval()
        with torch.no_grad():
            y_pred_x, _ = self.forward(X, Z)  # only take φ branch output
            preds = (y_pred_x >= 0.5).float().cpu().numpy()
            y_np = y.cpu().numpy()

            # Precision / Recall / F1 for each class
            p, r, f, _ = precision_recall_fscore_support(y_np, preds, zero_division=0)

            # AUC‐ROC
            try:
                auc = roc_auc_score(y_np, y_pred_x.cpu().numpy())
            except ValueError:
                auc = 0.0

        return {
            "auc": auc,
            "p0": float(p[0]), "r0": float(r[0]), "f0": float(f[0]),
            "p1": float(p[1]), "r1": float(r[1]), "f1": float(f[1])
        }


################################################################################
# 3) Simultaneous Multi‐View + Multi‐Task Model
################################################################################
class SimultaneousMultiViewMultiTaskNN(nn.Module):
    """
    Multi‐view + multi‐task with shared head ψ:

    Architecture:
      - φ(x): encoder for view1 → hidden_x
      - β(z): encoder for view2 → hidden_z
      - ψ: shared network producing 4 outputs per example:
          [m_x, aux1_pred, aux2_pred, aux3_pred]
      - m_x = main prediction from view1
      - mz  = main prediction from view2
      - a1, a2, a3 = three auxiliary outputs from view1

    Loss:
      L = BCE(mx, y) + BCE(mz, y)
          + λ_aux * (MSE(a1, aux1_target) + MSE(a2, aux2_target) + BCE(a3, aux3_target))
          + λ_direct * MSE(hidden_x, hidden_z)
    """

    def __init__(self, dim_x: int, dim_z: int, hidden: int = 64,
                 lr: float = 1e-3, epochs: int = 100,
                 lambda_aux: float = 0.3, lambda_direct: float = 0.3):
        """
        Args:
            dim_x (int): Dimension of view1 features.
            dim_z (int): Dimension of view2 features.
            hidden (int): Hidden dimension for both encoders.
            lr (float): Learning rate.
            epochs (int): Number of epochs.
            lambda_aux (float): Weight for auxiliary losses.
            lambda_direct (float): Weight for direct alignment loss between hidden_x and hidden_z.
        """
        super().__init__()
        self.epochs = epochs
        self.lambda_aux = lambda_aux
        self.lambda_direct = lambda_direct

        # Encoder for view1: φ(x) → hidden_x
        self.encoder_x = nn.Sequential(
            nn.Linear(dim_x, hidden),
            nn.ReLU()
        )

        # Encoder for view2: β(z) → hidden_z
        self.encoder_z = nn.Sequential(
            nn.Linear(dim_z, hidden),
            nn.ReLU()
        )

        # Shared head ψ: maps hidden → 4 outputs
        #   out[:,0] = main prediction mx
        #   out[:,1] = aux1 prediction
        #   out[:,2] = aux2 prediction
        #   out[:,3] = aux3 prediction
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)
        )

        # Loss functions
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

        # Optimizer
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Forward pass:
          - Encode x → s
          - Encode z → sp
          - Pass both s and sp through shared head ψ
          - Extract predictions:
              mx, aux1, aux2, aux3  from ψ(s)
              mz  from ψ(sp) (only use first output index)
        Returns:
            s (Tensor [batch, hidden]),
            sp (Tensor [batch, hidden]),
            mx (Tensor [batch]),
            mz (Tensor [batch]),
            aux1, aux2, aux3 (Tensor [batch]) from φ branch
        """
        s = self.encoder_x(x)   # [batch, hidden]
        sp = self.encoder_z(z)  # [batch, hidden]

        out_x = self.shared(s)   # [batch, 4]
        out_z = self.shared(sp)  # [batch, 4]

        mx = torch.sigmoid(out_x[:, 0])   # main from view1
        a1 = out_x[:, 1]                  # auxiliary regression 1
        a2 = out_x[:, 2]                  # auxiliary regression 2
        a3 = torch.sigmoid(out_x[:, 3])   # auxiliary classification 3

        mz = torch.sigmoid(out_z[:, 0])   # main from view2

        return s, sp, mx, mz, a1, a2, a3

    def compute_loss(self,
                     x: torch.Tensor,
                     z: torch.Tensor,
                     y: torch.Tensor,
                     a1_target: torch.Tensor,
                     a2_target: torch.Tensor,
                     a3_target: torch.Tensor):
        """
        Compute combined loss on a batch:
          - L_main = BCE(mx, y) + BCE(mz, y)
          - L_aux  = MSE(a1_pred, a1_target) + MSE(a2_pred, a2_target) + BCE(a3_pred, a3_target)
          - L_direct = MSE(s, sp)  (alignment of hidden representations)
          - Total L = L_main + λ_aux * L_aux + λ_direct * L_direct
        """
        s, sp, mx, mz, a1_pred, a2_pred, a3_pred = self.forward(x, z)

        # Main losses
        main_loss = self.bce(mx, y.view(-1)) + self.bce(mz, y.view(-1))

        # Auxiliary losses
        aux_loss = self.mse(a1_pred, a1_target.view(-1)) \
                 + self.mse(a2_pred, a2_target.view(-1)) \
                 + self.bce(a3_pred, a3_target.view(-1))

        # Direct view alignment loss
        direct_loss = self.mse(s, sp)

        return main_loss + self.lambda_aux * aux_loss + self.lambda_direct * direct_loss

    def train_model(self,
                    X: torch.Tensor,
                    Z: torch.Tensor,
                    y: torch.Tensor,
                    a1: torch.Tensor,
                    a2: torch.Tensor,
                    a3: torch.Tensor,
                    x_val: torch.Tensor = None,
                    z_val: torch.Tensor = None,
                    y_val: torch.Tensor = None,
                    a1_val: torch.Tensor = None,
                    a2_val: torch.Tensor = None,
                    a3_val: torch.Tensor = None,
                    record_loss: bool = False,
                    early_stopping_patience: int = 10):
        """
        Train the multi‐view + multi‐task network.

        Args:
            X, Z, y, a1, a2, a3 (Tensor): Training features and targets for main and auxiliary tasks.
            x_val, z_val, y_val, a1_val, a2_val, a3_val (Tensor): Optional validation data.
            record_loss (bool): If True, returns (train_losses, val_losses).
            early_stopping_patience (int): Patience for early stopping on validation loss.

        Returns:
            (train_losses, val_losses) if record_loss else None
        """
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for _ in range(self.epochs):
            self.train()
            self.opt.zero_grad()

            loss = self.compute_loss(X, Z, y, a1, a2, a3)
            loss.backward()
            self.opt.step()

            if record_loss:
                train_losses.append(loss.item())

            # Early stopping on validation data
            if x_val is not None and y_val is not None and z_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = self.compute_loss(x_val, z_val, y_val, a1_val, a2_val, a3_val)
                if record_loss:
                    val_losses.append(val_loss.item())

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break

        if record_loss:
            return train_losses, val_losses

    def evaluate(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 Z: torch.Tensor):
        """
        Evaluate model on test set: compute accuracy, per-class accuracy, AUC, and per-class P/R/F.

        Args:
            X (Tensor): View1 test features.
            y (Tensor): Test labels.
            Z (Tensor): View2 test features (zeros for test).

        Returns:
            Dict with keys: 
              "accuracy", "accuracy_class0", "accuracy_class1",
              "auc", "p0", "r0", "f0", "p1", "r1", "f1"
        """
        self.eval()
        with torch.no_grad():
            _, _, mx, mz, _, _, _ = self.forward(X, Z)
            pred = (mx + mz) / 2.0  # average predictions from both views

        y_true = y.view(-1).cpu().numpy()
        y_score = pred.cpu().numpy()
        y_hat = (y_score >= 0.5).astype(float)

        # Overall accuracy
        acc = (y_hat == y_true).mean()
        # Class‐specific accuracies
        acc0 = (y_hat[y_true == 0] == 0).mean() if (y_true == 0).any() else 0.0
        acc1 = (y_hat[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0

        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

        p, r, f, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0)

        return {
            "accuracy": acc,
            "accuracy_class0": acc0,
            "accuracy_class1": acc1,
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
        }


################################################################################
# 4) Simple Multi‐View MLP (view alignment loss)
################################################################################
class MultiViewMLP(nn.Module):
    """
    φ(x) branch vs β(z) branch, enforce latent alignment:

      L = BCE(ψ(φ(x)), y) + λ * MSE( φ(x), β(z) )

    - φ: MLP mapping X → hidden_x
    - β: MLP mapping Z → hidden_z
    - ψ: classifier mapping hidden_x → probability
    """

    def __init__(self,
                 dim_x: int,
                 dim_z: int,
                 hidden: int = 32,
                 num_layers_x: int = 3,
                 num_layers_z: int = 2,
                 lr: float = 1e-3,
                 lambda_view: float = 0.3,
                 epochs: int = 100):
        """
        Args:
            dim_x (int): Dimension of view1 features.
            dim_z (int): Dimension of view2 features.
            hidden (int): Hidden dimension for both branches.
            num_layers_x (int): Number of layers in φ.
            num_layers_z (int): Number of layers in β.
            lr (float): Learning rate.
            lambda_view (float): Weight for view alignment MSE loss.
            epochs (int): Number of training epochs.
        """
        super().__init__()
        self.epochs = epochs
        self.lambda_view = lambda_view

        # Build φ: X → hidden (num_layers_x)
        layers_x = []
        in_dim = dim_x
        for _ in range(num_layers_x):
            layers_x += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        self.phi = nn.Sequential(*layers_x)

        # Build β: Z → hidden (num_layers_z)
        layers_z = []
        in_dim = dim_z
        for _ in range(num_layers_z):
            layers_z += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        self.beta = nn.Sequential(*layers_z)

        # Classifier ψ: hidden → scalar probability
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

        # Optimizer for entire network
        self.opt = optim.Adam(self.parameters(), lr=lr)

        # Loss functions
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Forward pass:
          - Compute φ(x) → hidden_x
          - Compute β(z) → hidden_z
          - Compute ψ(hidden_x) → y_hat
        Returns:
            y_hat (Tensor [batch]), hidden_x (Tensor [batch, hidden]), hidden_z (Tensor [batch, hidden])
        """
        hidden_x = self.phi(x)
        hidden_z = self.beta(z)
        y_hat = self.classifier(hidden_x).view(-1)
        return y_hat, hidden_x, hidden_z

    def train_model(self,
                    X: torch.Tensor,
                    Z: torch.Tensor,
                    y: torch.Tensor,
                    x_val: torch.Tensor = None,
                    y_val: torch.Tensor = None,
                    z_val: torch.Tensor = None,
                    record_loss: bool = False,
                    early_stopping_patience: int = 10):
        """
        Train φ + β + ψ with view alignment loss.

        Args:
            X (Tensor): View1 training features.
            Z (Tensor): View2 training features.
            y (Tensor): Training labels.
            x_val, y_val, z_val (Tensor): Optional validation data for early stopping.
            record_loss (bool): If True, returns (train_losses, val_losses).
            early_stopping_patience (int): Patience for early stopping on validation BCE loss.

        Returns:
            (train_losses, val_losses) if record_loss else None
        """
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for _ in range(self.epochs):
            self.train()
            self.opt.zero_grad()

            y_pred, hidden_x, hidden_z = self.forward(X, Z)
            loss_main = self.bce(y_pred, y.view(-1))
            loss_align = self.mse(hidden_x, hidden_z)
            loss = loss_main + self.lambda_view * loss_align

            loss.backward()
            self.opt.step()

            if record_loss:
                train_losses.append(loss_main.item())

            # Early stopping using validation BCE only
            if x_val is not None and y_val is not None and z_val is not None:
                self.eval()
                with torch.no_grad():
                    y_val_pred, _, _ = self.forward(x_val, z_val)
                    val_loss_main = self.bce(y_val_pred, y_val.view(-1))
                if record_loss:
                    val_losses.append(val_loss_main.item())

                if val_loss_main.item() < best_val_loss:
                    best_val_loss = val_loss_main.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break

        if record_loss:
            return train_losses, val_losses

    def evaluate(self, X: torch.Tensor, y: torch.Tensor, Z: torch.Tensor = None):
        """
        Evaluate model on test set: compute AUC and per-class P/R/F.

        Args:
            X (Tensor): View1 test features.
            y (Tensor): Test labels.
            Z (Tensor): View2 test features (zeros for test).

        Returns:
            Dict with keys: "auc", "p0", "r0", "f0", "p1", "r1", "f1"
        """
        self.eval()
        with torch.no_grad():
            if Z is None:
                # If Z is None, feed a zero tensor of matching shape
                Z_dummy = torch.zeros_like(X)
                y_pred, _, _ = self.forward(X, Z_dummy)
            else:
                y_pred, _, _ = self.forward(X, Z)

        y_true = y.view(-1).cpu().numpy()
        y_score = y_pred.cpu().numpy()
        y_hat = (y_score >= 0.5).astype(float)

        p, r, f, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = 0.0

        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
        }

def main():
    """
    Main entry point to run Multi‐View experiments for:
      1) Two‐Loss pretrain + finetune model (MultiViewNN_TwoLoss)
      2) Simultaneous two‐loss model (MultiViewNN_Simul)

    Steps:
      - Set random seed for reproducibility.
      - Create (or overwrite) results CSV with header.
      - Load and preprocess data into (grid, train, test) splits.
      - For each model:
          * Perform hyperparameter grid search on the balanced grid set.
          * Run multiple repeated experiments on the train pool with best params.
          * Aggregate per‐run metrics and write to CSV.
          * Print final aggregated results.
    """
    print("Original / 70B")
    # -----------------------
    # 1) Ensure reproducibility
    # -----------------------
    set_seed(42)

    # -----------------------
    # 2) Prepare output CSV
    # -----------------------
    OUTPUT_CSV = "multiview_results.csv"
    # If a previous results file exists, remove it
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    # Write CSV header: Method, AUC, Params
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "AUC", "Params"])

    # -----------------------
    # 3) Load and preprocess data
    # -----------------------
    # Instantiate preprocessor with path to augmented CSV and UCI dataset ID
    dp = MultiViewDatasetPreprocessor(
        side_info_path="prompting/augmented_data_70B.csv",
        dataset_id=891
    )
    # Returns:
    #   grid_X, grid_y, grid_Z for hyperparameter tuning (balanced grid)
    #   train_X, train_y, train_Z for repeated experiments (train pool)
    #   test_X, test_y, test_Z for final evaluation (test pool)
    (grid_X, grid_y, grid_Z), \
    (train_X, train_y, train_Z), \
    (test_X, test_y, test_Z) = dp.preprocess()

    # -----------------------
    # 4) Define experiment configurations
    # -----------------------
    # We will run two Multi‐View models:
    #   - "TwoLoss": Two‐Loss pretrain + finetune (MultiViewNN_TwoLoss)
    #   - "Simul": Simultaneous two‐loss (MultiViewNN_Simul)
    experiments = [
        {
            "name": "TwoLoss",
            "model": MultiViewNN_TwoLoss,
            "param_grid": {
                "hidden": [32, 64, 128, 256],
                "num_layers_x": [1, 2, 3, 4],
                "num_layers_z": [1, 2, 3, 4],
                "lr": [0.001, 0.01],
                "epochs": [100],
                "lambda_aux": [0.3, 0.5]
            }
        },
        # {
        #     "name": "Simul",
        #     "model": MultiViewNN_Simul,
        #     "param_grid": {
        #         "hidden": [32, 64],
        #         "lr": [0.001, 0.01],
        #         "num_layers_x": [1, 2, 3, 4],
        #         "num_layers_z": [1, 2, 3, 4],
        #         "epochs": [100],
        #         "lambda_aux": [0.3, 0.5]
        #     }
        # },
        # # {
        # #     "name": "MT-MLP-Simul",
        # #     "model": SimultaneousMultiViewMultiTaskNN,
        # #     "param_grid": {
        # #         "hidden": [64],
        # #         "lr": [0.001, 0.01],
        # #         "epochs": [100],
        # #         "lambda_aux": [0.3, 0.5],
        # #         "lambda_direct": [0.3, 0.5]
        # #     }
        # # },
        # {
        #     "name": "MV-MLP",
        #     "model": MultiViewMLP,
        #     "param_grid": {
        #         "hidden": [32, 64],
        #         "num_layers_x": [2, 3],
        #         "num_layers_z": [1, 2],
        #         "lr": [0.001, 0.01],
        #         "lambda_view": [0.3, 0.5],
        #         "epochs": [100]
        #     }
        # }
    ]

    # -----------------------
    # 5) Loop over each experiment
    # -----------------------
    for cfg in experiments:
        name = cfg["name"]
        ModelClass = cfg["model"]
        param_grid = cfg["param_grid"]

        # (a) Hyperparameter grid search on the balanced grid set
        #     Uses ROC AUC on the main task for validation
        best_params, best_auc = grid_search_multiview(
            model_class=ModelClass,
            param_grid=param_grid,
            X=grid_X,
            Z=grid_Z,
            y=grid_y,
            cv=3
        )
        print(f"[{name}] Best params: {best_params}, CV AUC: {best_auc:.4f}")

        # (b) Run repeated experiments on the training pool with the best hyperparameters
        #     Each run:
        #       - Balanced sampling of up to 250 examples per class from train pool
        #       - 80/20 train/validation split
        #       - Train model (pretrain if applicable) with loss recording
        #       - Evaluate on a random 125‐subset of the test pool
        results = run_experiments_multiview(
            model_class=ModelClass,
            method_name=name,
            num_runs=10,
            best_params=best_params,
            train_X=train_X,
            train_Z=train_Z,
            train_y=train_y,
            test_X=test_X,
            test_Z=test_Z,
            test_y=test_y
        )

        # (c) Aggregate metrics across runs (mean ± std for each metric)
        aggregated = aggregate_metrics(results)

        # (d) Append a row to the CSV: [Method, AUC, Params]
        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, aggregated.get("auc", ""), str(best_params)])

        # (e) Print final aggregated results to console
        print(f"\n=== Final Aggregated Results for {name} ===")
        print(f"AUC (mean ± std): {aggregated.get('auc', '')}")
        print(f"Best Hyperparameters: {best_params}")
        print("---------------------------------------------------")
        # Optionally print all aggregated metrics (precision/recall/F1 etc.)
        print_aggregated_results(name, aggregated, str(best_params))


if __name__ == "__main__":
    main()


# import os
# import random
# import csv
# import itertools

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import inspect

# from ucimlrepo import fetch_ucirepo
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import KFold
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
# import copy


# # =============================================================================
# #                               UTILITY FUNCTIONS
# # =============================================================================

# def set_seed(seed: int):
#     """
#     Fix random seed for reproducibility across random, NumPy, and PyTorch.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def plot_loss_curve(train_losses: list, val_losses: list, filename: str):
#     """
#     Plot and save training vs. validation loss curve to a file.
#     """
#     plt.figure(figsize=(8, 6))
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(val_losses, label="Validation Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Train vs Validation Loss")
#     plt.legend()

#     folder = os.path.dirname(filename)
#     if folder and not os.path.exists(folder):
#         os.makedirs(folder, exist_ok=True)

#     plt.savefig(filename)
#     plt.close()

# def run_experiments_multiview(
#     model_class,
#     method_name: str,
#     num_runs: int,
#     param_grid: dict,
#     train_X_pool: torch.Tensor,
#     train_Z_pool: torch.Tensor,
#     train_y_pool: torch.Tensor,
#     test_X_pool: torch.Tensor,
#     test_Z_pool: torch.Tensor,
#     test_y_pool: torch.Tensor,
#     cv: int = 3
# ) -> (list, list):
#     """
#     Combined function that, for each iteration:
#       1. Samples 250 positives + 250 negatives *without replacement* from train_pool → 500 train samples.
#       2. Samples 125 test examples (may overlap across iterations) from test_pool.
#       3. Performs 3-fold CV on those 500 train samples (using both X and Z) to find best hyperparameters.
#       4. Retrains the model on all 500 train samples using best hyperparameters.
#       5. Evaluates the retrained model on the 125-sample test set → collects metrics.
#       6. Repeats num_runs times, ensuring that each 500-sample training set does not overlap across iterations.

#     Args:
#         model_class: Multi-view model class implementing __init__(dim_x, dim_z, **params),
#                      train_model(X, Z, y, x_val=None, z_val=None, y_val=None, record_loss=False), 
#                      optionally pretrain_aux(Z, y), and evaluate(X, y, Z).
#         method_name: String name for printing/logging.
#         num_runs: Number of disjoint experiments.
#         param_grid: Dict mapping hyperparameter names to lists of candidate values.
#         train_X_pool: (N_train, dim_x) tensor of view1 features.
#         train_Z_pool: (N_train, dim_z) tensor of view2 features.
#         train_y_pool: (N_train,) tensor of binary labels.
#         test_X_pool:  (N_test, dim_x) tensor of view1 features (test pool).
#         test_Z_pool:  (N_test, dim_z) tensor of view2 features (test pool).
#         test_y_pool:  (N_test,) tensor of binary labels (test pool).
#         cv: Number of folds for internal grid-search CV.

#     Returns:
#         metrics_list: List[dict] of evaluation metrics (main task) per iteration.
#         best_params_list: List[dict] of best hyperparameter dict found in each iteration.
#     """

#     metrics_list = []
#     best_params_list = []

#     # Build a DataFrame to identify positive/negative indices easily
#     df_pool = pd.DataFrame(train_X_pool.numpy())
#     df_pool['Diabetes_binary'] = train_y_pool.numpy()
#     class1_indices = df_pool[df_pool['Diabetes_binary'] == 1].index.tolist()
#     class0_indices = df_pool[df_pool['Diabetes_binary'] == 0].index.tolist()

#     used_indices = set()
#     test_pool_size = len(test_X_pool)

#     # Inspect model __init__ to know which hyperparameters to pass
#     sig = inspect.signature(model_class.__init__).parameters

#     for run_idx in range(1, num_runs + 1):
#         # ---------------------------------------
#         # 1) Balanced sampling: 250 positives + 250 negatives, no overlap
#         # ---------------------------------------
#         available_pos = [i for i in class1_indices if i not in used_indices]
#         available_neg = [i for i in class0_indices if i not in used_indices]
#         sampled_pos = random.sample(available_pos, 250)
#         sampled_neg = random.sample(available_neg, 250)
#         train_idx = sampled_pos + sampled_neg
#         used_indices.update(train_idx)

#         x_train = train_X_pool[train_idx]   # shape [500, dim_x]
#         z_train = train_Z_pool[train_idx]   # shape [500, dim_z]
#         y_train = train_y_pool[train_idx]   # shape [500]

#         # ---------------------------------------
#         # 2) Sample 125 test examples (may overlap across runs)
#         # ---------------------------------------
#         test_idx = random.sample(range(test_pool_size), 125)
#         x_test = test_X_pool[test_idx]      # shape [125, dim_x]
#         z_test = test_Z_pool[test_idx]      # shape [125, dim_z]
#         y_test = test_y_pool[test_idx]      # shape [125]

#         # ---------------------------------------
#         # 3) 3-fold CV for hyperparameter search
#         # ---------------------------------------
#         best_auc = -1.0
#         best_params = None
#         keys = list(param_grid.keys())

#         X_np = x_train.numpy()
#         Z_np = z_train.numpy()
#         y_np = y_train.numpy()

#         kf = KFold(n_splits=cv, shuffle=True, random_state=42)
#         for combo in itertools.product(*(param_grid[k] for k in keys)):
#             params = dict(zip(keys, combo))
#             fold_aucs = []

#             for tr_idx_cv, val_idx_cv in kf.split(X_np):
#                 # Build CV training/validation splits
#                 x_tr_cv = torch.tensor(X_np[tr_idx_cv], dtype=torch.float32)
#                 z_tr_cv = torch.tensor(Z_np[tr_idx_cv], dtype=torch.float32)
#                 y_tr_cv = torch.tensor(y_np[tr_idx_cv], dtype=torch.float32)

#                 x_va_cv = torch.tensor(X_np[val_idx_cv], dtype=torch.float32)
#                 z_va_cv = torch.tensor(Z_np[val_idx_cv], dtype=torch.float32)
#                 y_va_cv = torch.tensor(y_np[val_idx_cv], dtype=torch.float32)

#                 # Instantiate model with hyperparameters
#                 init_kwargs = {}
#                 # Always supply dim_x and dim_z if required
#                 if 'dim_x' in sig:
#                     init_kwargs['dim_x'] = X_np.shape[1]
#                 if 'dim_z' in sig:
#                     init_kwargs['dim_z'] = Z_np.shape[1]
#                 for k, v in params.items():
#                     if k in sig:
#                         init_kwargs[k] = v

#                 model = model_class(**init_kwargs)

#                 # Pretrain auxiliary branch if method exists
#                 if hasattr(model, "pretrain_aux"):
#                     model.pretrain_aux(z_tr_cv, y_tr_cv)

#                 # Train on CV‐training fold
#                 model.train_model(
#                     x_tr_cv, z_tr_cv, y_tr_cv,
#                     x_val=x_va_cv, z_val=z_va_cv, y_val=y_va_cv,
#                     record_loss=False
#                 )

#                 # Validate: compute AUC on φ‐branch output for the validation fold
#                 model.eval()
#                 with torch.no_grad():
#                     y_pred_va, _ = model.forward(x_va_cv, z_va_cv)
#                 try:
#                     auc_score = roc_auc_score(y_va_cv.cpu().numpy(),
#                                               y_pred_va.cpu().numpy())
#                 except ValueError:
#                     auc_score = 0.0
#                 fold_aucs.append(auc_score)

#             mean_auc = float(np.mean(fold_aucs))
#             if mean_auc > best_auc:
#                 best_auc = mean_auc
#                 best_params = params.copy()

#         print(f"[{method_name}] Iteration {run_idx} → Best CV params: {best_params}, Avg CV AUC: {best_auc:.4f}")
#         best_params_list.append(best_params)

#         # ---------------------------------------
#         # 4) Retrain on all 500 samples with best_params
#         # ---------------------------------------
#         init_kwargs = {}
#         if 'dim_x' in sig:
#             init_kwargs['dim_x'] = x_train.shape[1]
#         if 'dim_z' in sig:
#             init_kwargs['dim_z'] = z_train.shape[1]
#         for k, v in best_params.items():
#             if k in sig:
#                 init_kwargs[k] = v

#         final_model = model_class(**init_kwargs)

#         # Pretrain auxiliary branch if available
#         if hasattr(final_model, "pretrain_aux"):
#             final_model.pretrain_aux(z_train, y_train)

#         # Train on full 500‐sample set
#         final_model.train_model(x_train, z_train, y_train, record_loss=False)

#         # ---------------------------------------
#         # 5) Evaluate on the 125‐sample test set
#         # ---------------------------------------
#         metrics = final_model.evaluate(x_test, y_test, z_test)
#         print(f"[{method_name}] Iteration {run_idx} → Test metrics: {metrics}")
#         metrics_list.append(metrics)

#     return metrics_list, best_params_list


# def aggregate_metrics(metrics_list: list):
#     """
#     Aggregate a list of metric dictionaries by computing mean ± std for each key.

#     Args:
#         metrics_list: list of dicts, all sharing the same keys.

#     Returns:
#         agg: dict mapping each key to a string "mean ± std".
#     """
#     agg = {}
#     for key in metrics_list[0].keys():
#         vals = [m[key] for m in metrics_list]
#         mean_val = np.mean(vals)
#         std_val = np.std(vals)
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

# def print_aggregated_results(method: str, metrics: dict, params: str):
#     """
#     Print the final aggregated results for a given method.

#     Args:
#         method: model name.
#         metrics: aggregated metrics dict (mean ± std).
#         params: string representation of hyperparameters.
#     """
#     print("\n=== Final Aggregated Results ===")
#     print(f"Method: {method}")
#     if "accuracy" in metrics:
#         print(f"Overall Accuracy: {metrics.get('accuracy', '')}")
#     if "accuracy_class0" in metrics:
#         print(f"Class0 Accuracy: {metrics.get('accuracy_class0', '')}")
#     if "accuracy_class1" in metrics:
#         print(f"Class1 Accuracy: {metrics.get('accuracy_class1', '')}")
#     print(f"AUC: {metrics.get('auc', '')}")
#     print(f"Class0 → Precision: {metrics.get('p0', '')}, Recall: {metrics.get('r0', '')}, F1: {metrics.get('f0', '')}")
#     print(f"Class1 → Precision: {metrics.get('p1', '')}, Recall: {metrics.get('r1', '')}, F1: {metrics.get('f1', '')}")
#     print(f"Best Params: {params}")
#     print("---------------------------------------------------")


# # =============================================================================
# #                         DATA PREPROCESSOR CLASS
# # =============================================================================

# class MultiViewDatasetPreprocessor:
#     """
#     Preprocessor for multi-view learning without one-hot encoding for view2 categorical features.

#     - During training: use augmented CSV (side info) for grid set & train pool.
#     - Test set is drawn from original UCI data; test's view2 is a zero tensor.
#     - Scale only 'BMI' among view1 features; leave other view1 features unchanged.
#     - Scale continuous view2 features and label-encode categorical view2 features.
#     """

#     def __init__(self, dataset_id: int = 891, side_info_path: str = "prompting/augmented_data.csv"):
#         """
#         Args:
#             dataset_id: UCI dataset ID.
#             side_info_path: path to LLM-augmented CSV.
#         """
#         # Fetch original UCI dataset
#         self.original_data = fetch_ucirepo(id=dataset_id)
#         self.X_original = self.original_data.data.features  # pandas DataFrame
#         self.y_original = self.original_data.data.targets   # pandas Series

#         # Side-information CSV path
#         self.side_info_path = side_info_path

#         # View1 columns (original clinical features)
#         self.view1_cols = [
#             "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
#             "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
#             "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
#             "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
#         ]
#         self.view1_scaling_cols = ["BMI"]

#         # View2 columns (LLM-augmented privileged info)
#         self.view2_cont_cols = [
#             "predict_hba1c", "predict_cholesterol", "systolic_bp",
#             "diastolic_bp", "exercise_freq", "hi_sugar_freq"
#         ]
#         self.view2_cat_cols = ["employment_status"]

#         # Main target column
#         self.target = "Diabetes_binary"

#          # Standardizer for continuous features
#         self.scaler_bmi = StandardScaler()
#         self.scaler_view2_cont = StandardScaler()
#         self.label_encoders = {col: LabelEncoder() for col in self.view2_cat_cols}


#     def preprocess(self):
#         """
#         Execute preprocessing and return:
#           - grid set: (grid_x, grid_y, grid_z)
#           - train pool: (train_x, train_y, train_z)
#           - test set: (test_x, test_y, test_z)
#         """
#         # 1) Load augmented CSV that includes view2 and target
#         augmented_df = pd.read_csv(self.side_info_path)

#         # 2) Identify missing values in view2 and target, print stats
#         cols_to_check = self.view2_cont_cols + self.view2_cat_cols + [self.target]
#         na_counts = augmented_df[cols_to_check].isna().sum()
#         na_ratio = (na_counts / len(augmented_df)) * 100
#         col_dtypes = augmented_df[cols_to_check].dtypes
#         for col in cols_to_check:
#             print(f"{col}: {na_counts[col]} missing ({na_ratio[col]:.2f}%) — dtype: {col_dtypes[col]}")

#         # 3) Fill missing values in continuous and categorical view2 and target with median
#         for col in cols_to_check:
#             if augmented_df[col].isna().any():
#                 median_value = augmented_df[col].median()
#                 augmented_df[col].fillna(median_value, inplace=True)

#         # 4) Create balanced 'grid' set (max 375 samples per class)
#         pos_idx = augmented_df[augmented_df[self.target] == 1].index.tolist()
#         neg_idx = augmented_df[augmented_df[self.target] == 0].index.tolist()
#         n_pos = min(len(pos_idx), 375)
#         n_neg = min(len(neg_idx), 375)
#         grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
#         grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
#         grid_df = pd.concat([grid_pos, grid_neg], ignore_index=True)

#         # Remaining rows form the train pool
#         train_pool = augmented_df.drop(index=grid_df.index, errors="ignore").reset_index(drop=True)

#         # 5) Construct test set from original UCI data (no privileged view2)
#         original_df = self.X_original.copy()
#         original_df[self.target] = self.y_original.values
#         test_pool_df = original_df.drop(index=augmented_df.index, errors="ignore").reset_index(drop=True)

#         # 6) Scale view1's BMI
#         self.scaler_bmi.fit(train_pool[["BMI"]])

#         def process_view1(df: pd.DataFrame) -> pd.DataFrame:
#             df_copy = df.copy()
#             df_copy["BMI"] = self.scaler_bmi.transform(df[["BMI"]])
#             return df_copy

#         grid_df_proc = process_view1(grid_df)
#         train_pool_df_proc = process_view1(train_pool)
#         test_pool_df_proc = process_view1(test_pool_df)

#         # 7) Scale continuous view2 features using train pool
#         self.scaler_view2_cont.fit(train_pool[self.view2_cont_cols])
#         grid_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(grid_df[self.view2_cont_cols])
#         train_pool[self.view2_cont_cols] = self.scaler_view2_cont.transform(train_pool[self.view2_cont_cols])

#         # 8) Label-encode categorical view2 features
#         for col in self.view2_cat_cols:
#             self.label_encoders[col].fit(augmented_df[col].astype(str))
#             grid_df[col] = self.label_encoders[col].transform(grid_df[col].astype(str))
#             train_pool[col] = self.label_encoders[col].transform(train_pool[col].astype(str))

#         # 9) Combine view2 continuous + categorical into a DataFrame, then to Tensor
#         grid_view2 = grid_df[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)
#         train_view2 = train_pool[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)

#         grid_z = torch.tensor(grid_view2.values, dtype=torch.float32)
#         train_z = torch.tensor(train_view2.values, dtype=torch.float32)
#         if grid_z.ndimension() == 1:
#             grid_z = grid_z.unsqueeze(1)
#             train_z = train_z.unsqueeze(1)

#         # 10) Convert view1 and target for grid and train to Tensors
#         def to_tensor(df: pd.DataFrame, feature_cols: list, target_col: str = None):
#             X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
#             y = torch.tensor(df[target_col].values, dtype=torch.float32).view(-1) if target_col else None
#             return X, y

#         grid_x, grid_y = to_tensor(grid_df_proc, self.view1_cols, self.target)
#         train_x, train_y = to_tensor(train_pool_df_proc, self.view1_cols, self.target)
#         test_x, test_y = to_tensor(test_pool_df_proc, self.view1_cols, self.target)

#         # 11) Test set's view2 is a zero tensor (no privileged info)
#         dim_view2 = grid_z.shape[1] if grid_z.ndimension() > 1 else 0
#         test_z = torch.zeros((test_x.shape[0], dim_view2), dtype=torch.float32)

#         return (grid_x, grid_y, grid_z), (train_x, train_y, train_z), (test_x, test_y, test_z)

# ################################################################################
# # 1) Two‐Loss Multi‐View Model
# ################################################################################
# class MultiViewNN_TwoLoss(nn.Module):
#     """
#     Two-branch multi-view model with independent depths:
#       - β: auxiliary branch on Z (privileged view)
#       - φ: main branch on X (primary view)
#       - ψ: shared classifier head

#     Training steps:
#       1) Pretrain β + ψ on (Z, y) to predict y from side information.
#       2) Freeze β and copy ψ to a fixed “teacher” classifier.
#       3) Train φ + ψ on (X, y) with auxiliary regularization:
#            L = BCE(ψ(φ(x)), y) + λ * BCE( teacher(β(z)), y )
#     """

#     def __init__(self, dim_x: int, dim_z: int, hidden: int = 32,
#                  num_layers_x: int = 2, num_layers_z: int = 1,
#                  lr: float = 1e-3, epochs: int = 100, lambda_aux: float = 0.5):
#         """
#         Args:
#             dim_x (int): Dimension of view1 features (X).
#             dim_z (int): Dimension of view2 features (Z).
#             hidden (int): Hidden dimension for both branches.
#             num_layers_x (int): Number of hidden layers in φ (main branch).
#             num_layers_z (int): Number of hidden layers in β (aux branch).
#             lr (float): Learning rate for optimizer.
#             epochs (int): Number of epochs for both pretraining and main training.
#             lambda_aux (float): Weight for auxiliary BCE loss term.
#         """
#         super().__init__()
#         self.epochs = epochs
#         self.lambda_aux = lambda_aux

#         # Build φ network (main branch on X)
#         self.phi = self._make_mlp(in_dim=dim_x, out_dim=hidden, n_layers=num_layers_x)

#         # Build β network (auxiliary branch on Z)
#         self.beta = self._make_mlp(in_dim=dim_z, out_dim=hidden, n_layers=num_layers_z)

#         # Shared classifier ψ (maps hidden → scalar probability)
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden, 1),
#             nn.Sigmoid()
#         )

#         # Optimizers: 
#         #   - aux_opt for pretraining β + ψ
#         #   - main_opt for training φ + ψ after pretraining
#         self.aux_opt = optim.Adam(
#             list(self.beta.parameters()) + list(self.classifier.parameters()),
#             lr=lr
#         )
#         self.main_opt = None

#         # Loss function
#         self.bce = nn.BCELoss()

#         # After pretraining, this holds a frozen copy of ψ
#         self.fixed_classifier = None

#     def _make_mlp(self, in_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
#         """
#         Helper to build an MLP of n_layers layers, each mapping → out_dim then ReLU.
#         """
#         layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
#         for _ in range(n_layers - 1):
#             layers += [nn.Linear(out_dim, out_dim), nn.ReLU()]
#         return nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor, z: torch.Tensor):
#         """
#         Forward pass:
#           - Compute φ(x) → hidden_x
#           - Compute β(z) → hidden_z
#           - Compute ψ(hidden_x) → y_x
#           - Compute ψ(hidden_z) → y_z
#         Returns:
#             y_x (Tensor [batch]), y_z (Tensor [batch])
#         """
#         hidden_x = self.phi(x)
#         hidden_z = self.beta(z)

#         y_x = self.classifier(hidden_x).view(-1)  # predictions from X branch
#         y_z = self.classifier(hidden_z).view(-1)  # predictions from Z branch
#         return y_x, y_z

#     def pretrain_aux(self, Z: torch.Tensor, y: torch.Tensor):
#         """
#         Pretrain auxiliary branch β + classifier ψ on (Z, y).
#         Freezes β afterward and creates a fixed copy of ψ to act as “teacher”.
#         """
#         # Create dummy input for φ (not used during aux pretraining)
#         dummy_x = torch.zeros(Z.size(0), self.phi[0].in_features, device=Z.device)

#         for _ in range(self.epochs):
#             self.aux_opt.zero_grad()
#             _, y_pred_z = self.forward(dummy_x, Z)
#             loss = self.bce(y_pred_z, y.view(-1))
#             loss.backward()
#             self.aux_opt.step()

#         # Freeze all parameters in β (aux branch)
#         for param in self.beta.parameters():
#             param.requires_grad = False

#         # Make a deep copy of classifier ψ as “teacher”
#         self.fixed_classifier = copy.deepcopy(self.classifier)

#         # Initialize main optimizer for φ + ψ
#         self.main_opt = optim.Adam(
#             list(self.phi.parameters()) + list(self.classifier.parameters()),
#             lr=self.aux_opt.defaults['lr']
#         )

#     def train_model(self,
#                     X: torch.Tensor,
#                     Z: torch.Tensor,
#                     y: torch.Tensor,
#                     x_val: torch.Tensor = None,
#                     z_val: torch.Tensor = None,
#                     y_val: torch.Tensor = None,
#                     record_loss: bool = False,
#                     early_stopping_patience: int = 10):
#         """
#         Train φ + ψ on X with auxiliary regularization from fixed β + ψ.

#         Args:
#             X (Tensor): View1 training features.
#             Z (Tensor): View2 training features.
#             y (Tensor): Labels for training.
#             x_val, z_val, y_val (Tensor): Optional validation sets for early stopping.
#             record_loss (bool): If True, returns (train_losses, val_losses).
#             early_stopping_patience (int): Patience for early stopping on validation BCE loss.

#         Returns:
#             (train_losses, val_losses) if record_loss else None
#         """
#         if self.fixed_classifier is None:
#             raise RuntimeError("Must call pretrain_aux() before training main branch.")

#         train_losses, val_losses = [], []
#         best_val_loss = float('inf')
#         patience_counter = 0

#         for _ in range(self.epochs):
#             self.train()
#             self.main_opt.zero_grad()

#             # Forward pass on training data
#             y_pred_x, _ = self.forward(X, Z)
#             with torch.no_grad():
#                 # Teacher predictions from fixed β + ψ
#                 y_teacher = self.fixed_classifier(self.beta(Z)).view(-1)

#             # Compute combined loss
#             loss_main = self.bce(y_pred_x, y.view(-1))
#             loss_aux = self.bce(y_pred_x, y_teacher)
#             loss = loss_main + self.lambda_aux * loss_aux

#             loss.backward()
#             self.main_opt.step()

#             if record_loss:
#                 train_losses.append(loss.item())

#             # Early stopping on validation set
#             if x_val is not None and y_val is not None and z_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     y_val_pred, _ = self.forward(x_val, z_val)
#                     val_loss = self.bce(y_val_pred, y_val.view(-1))
#                 if record_loss:
#                     val_losses.append(val_loss.item())

#                 # Check improvement
#                 if val_loss.item() < best_val_loss:
#                     best_val_loss = val_loss.item()
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= early_stopping_patience:
#                         break

#         if record_loss:
#             return train_losses, val_losses

#     def evaluate(self, X: torch.Tensor, y: torch.Tensor, Z: torch.Tensor):
#         """
#         Evaluate model on test set: compute AUC and per-class precision/recall/F1.

#         Args:
#             X (Tensor): View1 test features.
#             y (Tensor): Test labels.
#             Z (Tensor): View2 test features (zeros for test).

#         Returns:
#             Dict with keys: "auc", "p0", "r0", "f0", "p1", "r1", "f1"
#         """
#         self.eval()
#         with torch.no_grad():
#             y_pred, _ = self.forward(X, Z)
#         y_true = y.view(-1).cpu().numpy()
#         y_score = y_pred.cpu().numpy()
#         y_hat = (y_score >= 0.5).astype(float)

#         # Compute per-class metrics
#         p, r, f, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0)
#         try:
#             auc = roc_auc_score(y_true, y_score)
#         except ValueError:
#             auc = 0.0

#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#         }


# ################################################################################
# # 2) Simultaneous Two‐Loss Multi‐View Model
# ################################################################################
# class MultiViewNN_Simul(nn.Module):
#     """
#     Two‐branch simultaneous multi‐view model:
#       L = BCE(ψ(φ(x)), y) + λ * BCE(ψ(β(z)), y)

#     Both φ (view1 on X) and β (view2 on Z) are MLPs of configurable depth.
#     """
#     def __init__(
#         self,
#         dim_x: int,
#         dim_z: int,
#         hidden: int = 32,
#         num_layers_x: int = 2,
#         num_layers_z: int = 1,
#         lr: float = 1e-3,
#         epochs: int = 100,
#         lambda_aux: float = 0.5
#     ):
#         """
#         Args:
#             dim_x (int): Input dimensionality of view1 (X).
#             dim_z (int): Input dimensionality of view2 (Z).
#             hidden (int): Hidden‐unit size for all intermediate layers.
#             num_layers_x (int): Number of hidden layers in φ (view1 branch).
#             num_layers_z (int): Number of hidden layers in β (view2 branch).
#             lr (float): Learning rate for Adam optimizer.
#             epochs (int): Total epochs to train.
#             lambda_aux (float): Weight of the auxiliary BCE loss from view2.
#         """
#         super().__init__()
#         self.epochs     = epochs
#         self.lambda_aux = lambda_aux

#         # Build φ (view1 → hidden) as an MLP with num_layers_x hidden layers
#         # First layer maps dim_x → hidden, then (num_layers_x−1) layers hidden→hidden
#         phi_layers = []
#         in_dim = dim_x
#         for i in range(num_layers_x):
#             phi_layers.append(nn.Linear(in_dim, hidden))
#             phi_layers.append(nn.ReLU())
#             in_dim = hidden
#         self.phi = nn.Sequential(*phi_layers)

#         # Build β (view2 → hidden) as an MLP with num_layers_z hidden layers
#         in_dim = dim_z
#         beta_layers = []
#         for i in range(num_layers_z):
#             beta_layers.append(nn.Linear(in_dim, hidden))
#             beta_layers.append(nn.ReLU())
#             in_dim = hidden
#         self.beta = nn.Sequential(*beta_layers)

#         # Shared classifier ψ: hidden → 1 (sigmoid)
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden, 1),
#             nn.Sigmoid()
#         )

#         # Single optimizer for all parameters (φ, β, ψ)
#         self.opt = optim.Adam(self.parameters(), lr=lr)

#         # BCE loss
#         self.bce = nn.BCELoss()

#     def forward(self, x: torch.Tensor, z: torch.Tensor):
#         """
#         Forward pass:
#           - φ(x) → hidden_x
#           - β(z) → hidden_z
#           - ψ(hidden_x) = ŷ_x
#           - ψ(hidden_z) = ŷ_z
#         Returns:
#             (ŷ_x_flat, ŷ_z_flat) both of shape [batch]
#         """
#         # Compute hidden representation from view1
#         hidden_x = self.phi(x)      # [batch, hidden]
#         # Compute hidden representation from view2
#         hidden_z = self.beta(z)     # [batch, hidden]

#         # Classifier produces scalar probabilities
#         y_x = self.classifier(hidden_x).view(-1)  # [batch]
#         y_z = self.classifier(hidden_z).view(-1)  # [batch]
#         return y_x, y_z

#     def train_model(
#         self,
#         X: torch.Tensor,
#         Z: torch.Tensor,
#         y: torch.Tensor,
#         x_val: torch.Tensor = None,
#         y_val: torch.Tensor = None,
#         z_val: torch.Tensor = None,
#         early_stopping_patience: int = 10,
#         record_loss: bool = False
#     ):
#         """
#         Train both branches (φ + β + ψ) jointly. Each step:
#           L = BCE(ψ(φ(X)), y) + λ_aux * BCE(ψ(β(Z)), y)

#         Args:
#             X (Tensor): [N, dim_x] training features for view1.
#             Z (Tensor): [N, dim_z] training features for view2.
#             y (Tensor): [N] binary labels (0/1).
#             x_val, z_val, y_val: optional validation splits.
#             early_stopping_patience (int): patience on validation BCE of φ‐branch.
#             record_loss (bool): if True, return (train_loss_history, val_loss_history).
#         Returns:
#             If record_loss: (train_loss_history: List[float], val_loss_history: List[float])
#             Else: None
#         """
#         train_losses = []
#         val_losses = []
#         best_val_loss = float('inf')
#         patience = 0

#         for epoch in range(self.epochs):
#             self.train()
#             self.opt.zero_grad()

#             # Forward pass on training batch
#             y_pred_x, y_pred_z = self.forward(X, Z)  # each [N]

#             # Compute combined loss
#             loss_x = self.bce(y_pred_x, y)
#             loss_z = self.bce(y_pred_z, y)
#             loss = loss_x + self.lambda_aux * loss_z

#             # Backprop + optimizer step
#             loss.backward()
#             self.opt.step()

#             if record_loss:
#                 train_losses.append(loss.item())

#             # Validation step for early stopping (only φ‐branch on X)
#             if x_val is not None and y_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     y_val_pred_x, _ = self.forward(x_val, z_val)
#                     val_loss = self.bce(y_val_pred_x, y_val)
#                 if record_loss:
#                     val_losses.append(val_loss.item())

#                 # Early stopping check
#                 if val_loss.item() < best_val_loss:
#                     best_val_loss = val_loss.item()
#                     patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping_patience:
#                         break

#         if record_loss:
#             return train_losses, val_losses

#     def evaluate(
#         self,
#         X: torch.Tensor,
#         y: torch.Tensor,
#         Z: torch.Tensor
#     ):
#         """
#         Evaluate on a held‐out set:
#           - use only the φ‐branch output ψ(φ(X)) for computing AUC & p/r/f.

#         Args:
#             X (Tensor): [N_test, dim_x]
#             y (Tensor): [N_test]
#             Z (Tensor): [N_test, dim_z] (unused for final φ‐output but must supply)
#         Returns:
#             Dict with keys: "auc", "p0","r0","f0","p1","r1","f1"
#         """
#         self.eval()
#         with torch.no_grad():
#             y_pred_x, _ = self.forward(X, Z)  # only take φ branch output
#             preds = (y_pred_x >= 0.5).float().cpu().numpy()
#             y_np = y.cpu().numpy()

#             # Precision / Recall / F1 for each class
#             p, r, f, _ = precision_recall_fscore_support(y_np, preds, zero_division=0)

#             # AUC‐ROC
#             try:
#                 auc = roc_auc_score(y_np, y_pred_x.cpu().numpy())
#             except ValueError:
#                 auc = 0.0

#         return {
#             "auc": auc,
#             "p0": float(p[0]), "r0": float(r[0]), "f0": float(f[0]),
#             "p1": float(p[1]), "r1": float(r[1]), "f1": float(f[1])
#         }


# ################################################################################
# # 3) Simultaneous Multi‐View + Multi‐Task Model
# ################################################################################
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

# # class SimultaneousMultiViewMultiTaskNN(nn.Module):
# #     """
# #     Multi‐view + multi‐task with shared head ψ:

# #     Architecture:
# #       - φ(x): encoder for view1 → hidden_x
# #       - β(z): encoder for view2 → hidden_z
# #       - ψ: shared network producing 4 outputs per example:
# #           [m_x, aux1_pred, aux2_pred, aux3_pred]
# #       - m_x = main prediction from view1
# #       - mz  = main prediction from view2
# #       - a1, a2, a3 = three auxiliary outputs from view1

# #     Loss:
# #       L = BCE(mx, y) + BCE(mz, y)
# #           + λ_aux * (MSE(a1, aux1_target) + MSE(a2, aux2_target) + BCE(a3, aux3_target))
# #           + λ_direct * MSE(hidden_x, hidden_z)
# #     """

# #     def __init__(self, dim_x: int, dim_z: int, hidden: int = 64,
# #                  lr: float = 1e-3, epochs: int = 100,
# #                  lambda_aux: float = 0.3, lambda_direct: float = 0.3):
# #         """
# #         Args:
# #             dim_x (int): Dimension of view1 features.
# #             dim_z (int): Dimension of view2 features.
# #             hidden (int): Hidden dimension for both encoders.
# #             lr (float): Learning rate.
# #             epochs (int): Number of epochs.
# #             lambda_aux (float): Weight for auxiliary losses.
# #             lambda_direct (float): Weight for direct alignment loss between hidden_x and hidden_z.
# #         """
# #         super().__init__()
# #         self.epochs = epochs
# #         self.lambda_aux = lambda_aux
# #         self.lambda_direct = lambda_direct

# #         # Encoder for view1: φ(x) → hidden_x
# #         self.encoder_x = nn.Sequential(
# #             nn.Linear(dim_x, hidden),
# #             nn.ReLU()
# #         )

# #         # Encoder for view2: β(z) → hidden_z
# #         self.encoder_z = nn.Sequential(
# #             nn.Linear(dim_z, hidden),
# #             nn.ReLU()
# #         )

# #         # Shared head ψ: maps hidden → 4 outputs
# #         #   out[:,0] = main prediction mx
# #         #   out[:,1] = aux1 prediction
# #         #   out[:,2] = aux2 prediction
# #         #   out[:,3] = aux3 prediction
# #         self.shared = nn.Sequential(
# #             nn.Linear(hidden, hidden),
# #             nn.ReLU(),
# #             nn.Linear(hidden, 4)
# #         )

# #         # Loss functions
# #         self.bce = nn.BCELoss()
# #         self.mse = nn.MSELoss()

# #         # Optimizer
# #         self.opt = optim.Adam(self.parameters(), lr=lr)

# #     def forward(self, x: torch.Tensor, z: torch.Tensor):
# #         """
# #         Forward pass:
# #           - Encode x → s
# #           - Encode z → sp
# #           - Pass both s and sp through shared head ψ
# #           - Extract predictions:
# #               mx, aux1, aux2, aux3  from ψ(s)
# #               mz  from ψ(sp) (only use first output index)
# #         Returns:
# #             s (Tensor [batch, hidden]),
# #             sp (Tensor [batch, hidden]),
# #             mx (Tensor [batch]),
# #             mz (Tensor [batch]),
# #             aux1, aux2, aux3 (Tensor [batch]) from φ branch
# #         """
# #         s = self.encoder_x(x)   # [batch, hidden]
# #         sp = self.encoder_z(z)  # [batch, hidden]

# #         out_x = self.shared(s)   # [batch, 4]
# #         out_z = self.shared(sp)  # [batch, 4]

# #         mx = torch.sigmoid(out_x[:, 0])   # main from view1
# #         a1 = out_x[:, 1]                  # auxiliary regression 1
# #         a2 = out_x[:, 2]                  # auxiliary regression 2
# #         a3 = torch.sigmoid(out_x[:, 3])   # auxiliary classification 3

# #         mz = torch.sigmoid(out_z[:, 0])   # main from view2

# #         return s, sp, mx, mz, a1, a2, a3

# #     def compute_loss(self,
# #                      x: torch.Tensor,
# #                      z: torch.Tensor,
# #                      y: torch.Tensor,
# #                      a1_target: torch.Tensor,
# #                      a2_target: torch.Tensor,
# #                      a3_target: torch.Tensor):
# #         """
# #         Compute combined loss on a batch:
# #           - L_main = BCE(mx, y) + BCE(mz, y)
# #           - L_aux  = MSE(a1_pred, a1_target) + MSE(a2_pred, a2_target) + BCE(a3_pred, a3_target)
# #           - L_direct = MSE(s, sp)  (alignment of hidden representations)
# #           - Total L = L_main + λ_aux * L_aux + λ_direct * L_direct
# #         """
# #         s, sp, mx, mz, a1_pred, a2_pred, a3_pred = self.forward(x, z)

# #         # Main losses
# #         main_loss = self.bce(mx, y.view(-1)) + self.bce(mz, y.view(-1))

# #         # Auxiliary losses
# #         aux_loss = self.mse(a1_pred, a1_target.view(-1)) \
# #                  + self.mse(a2_pred, a2_target.view(-1)) \
# #                  + self.bce(a3_pred, a3_target.view(-1))

# #         # Direct view alignment loss
# #         direct_loss = self.mse(s, sp)

# #         return main_loss + self.lambda_aux * aux_loss + self.lambda_direct * direct_loss

# #     def train_model(self,
# #                     X: torch.Tensor,
# #                     Z: torch.Tensor,
# #                     y: torch.Tensor,
# #                     a1: torch.Tensor,
# #                     a2: torch.Tensor,
# #                     a3: torch.Tensor,
# #                     x_val: torch.Tensor = None,
# #                     z_val: torch.Tensor = None,
# #                     y_val: torch.Tensor = None,
# #                     a1_val: torch.Tensor = None,
# #                     a2_val: torch.Tensor = None,
# #                     a3_val: torch.Tensor = None,
# #                     record_loss: bool = False,
# #                     early_stopping_patience: int = 10):
# #         """
# #         Train the multi‐view + multi‐task network.

# #         Args:
# #             X, Z, y, a1, a2, a3 (Tensor): Training features and targets for main and auxiliary tasks.
# #             x_val, z_val, y_val, a1_val, a2_val, a3_val (Tensor): Optional validation data.
# #             record_loss (bool): If True, returns (train_losses, val_losses).
# #             early_stopping_patience (int): Patience for early stopping on validation loss.

# #         Returns:
# #             (train_losses, val_losses) if record_loss else None
# #         """
# #         train_losses, val_losses = [], []
# #         best_val_loss = float('inf')
# #         patience_counter = 0

# #         for _ in range(self.epochs):
# #             self.train()
# #             self.opt.zero_grad()

# #             loss = self.compute_loss(X, Z, y, a1, a2, a3)
# #             loss.backward()
# #             self.opt.step()

# #             if record_loss:
# #                 train_losses.append(loss.item())

# #             # Early stopping on validation data
# #             if x_val is not None and y_val is not None and z_val is not None:
# #                 self.eval()
# #                 with torch.no_grad():
# #                     val_loss = self.compute_loss(x_val, z_val, y_val, a1_val, a2_val, a3_val)
# #                 if record_loss:
# #                     val_losses.append(val_loss.item())

# #                 if val_loss.item() < best_val_loss:
# #                     best_val_loss = val_loss.item()
# #                     patience_counter = 0
# #                 else:
# #                     patience_counter += 1
# #                     if patience_counter >= early_stopping_patience:
# #                         break

# #         if record_loss:
# #             return train_losses, val_losses

# #     def evaluate(self,
# #                  X: torch.Tensor,
# #                  y: torch.Tensor,
# #                  Z: torch.Tensor):
# #         """
# #         Evaluate model on test set: compute accuracy, per-class accuracy, AUC, and per-class P/R/F.

# #         Args:
# #             X (Tensor): View1 test features.
# #             y (Tensor): Test labels.
# #             Z (Tensor): View2 test features (zeros for test).

# #         Returns:
# #             Dict with keys: 
# #               "accuracy", "accuracy_class0", "accuracy_class1",
# #               "auc", "p0", "r0", "f0", "p1", "r1", "f1"
# #         """
# #         self.eval()
# #         with torch.no_grad():
# #             _, _, mx, mz, _, _, _ = self.forward(X, Z)
# #             pred = (mx + mz) / 2.0  # average predictions from both views

# #         y_true = y.view(-1).cpu().numpy()
# #         y_score = pred.cpu().numpy()
# #         y_hat = (y_score >= 0.5).astype(float)

# #         # Overall accuracy
# #         acc = (y_hat == y_true).mean()
# #         # Class‐specific accuracies
# #         acc0 = (y_hat[y_true == 0] == 0).mean() if (y_true == 0).any() else 0.0
# #         acc1 = (y_hat[y_true == 1] == 1).mean() if (y_true == 1).any() else 0.0

# #         try:
# #             auc = roc_auc_score(y_true, y_score)
# #         except ValueError:
# #             auc = 0.0

# #         p, r, f, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0)

# #         return {
# #             "accuracy": acc,
# #             "accuracy_class0": acc0,
# #             "accuracy_class1": acc1,
# #             "auc": auc,
# #             "p0": p[0], "r0": r[0], "f0": f[0],
# #             "p1": p[1], "r1": r[1], "f1": f[1],
# #         }


# ################################################################################
# # 4) Simple Multi‐View MLP (view alignment loss)
# ################################################################################
# class MultiViewMLP(nn.Module):
#     """
#     φ(x) branch vs β(z) branch, enforce latent alignment:

#       L = BCE(ψ(φ(x)), y) + λ * MSE( φ(x), β(z) )

#     - φ: MLP mapping X → hidden_x
#     - β: MLP mapping Z → hidden_z
#     - ψ: classifier mapping hidden_x → probability
#     """

#     def __init__(self,
#                  dim_x: int,
#                  dim_z: int,
#                  hidden: int = 32,
#                  num_layers_x: int = 3,
#                  num_layers_z: int = 2,
#                  lr: float = 1e-3,
#                  lambda_view: float = 0.3,
#                  epochs: int = 100):
#         """
#         Args:
#             dim_x (int): Dimension of view1 features.
#             dim_z (int): Dimension of view2 features.
#             hidden (int): Hidden dimension for both branches.
#             num_layers_x (int): Number of layers in φ.
#             num_layers_z (int): Number of layers in β.
#             lr (float): Learning rate.
#             lambda_view (float): Weight for view alignment MSE loss.
#             epochs (int): Number of training epochs.
#         """
#         super().__init__()
#         self.epochs = epochs
#         self.lambda_view = lambda_view

#         # Build φ: X → hidden (num_layers_x)
#         layers_x = []
#         in_dim = dim_x
#         for _ in range(num_layers_x):
#             layers_x += [nn.Linear(in_dim, hidden), nn.ReLU()]
#             in_dim = hidden
#         self.phi = nn.Sequential(*layers_x)

#         # Build β: Z → hidden (num_layers_z)
#         layers_z = []
#         in_dim = dim_z
#         for _ in range(num_layers_z):
#             layers_z += [nn.Linear(in_dim, hidden), nn.ReLU()]
#             in_dim = hidden
#         self.beta = nn.Sequential(*layers_z)

#         # Classifier ψ: hidden → scalar probability
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden, 1),
#             nn.Sigmoid()
#         )

#         # Optimizer for entire network
#         self.opt = optim.Adam(self.parameters(), lr=lr)

#         # Loss functions
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def forward(self, x: torch.Tensor, z: torch.Tensor):
#         """
#         Forward pass:
#           - Compute φ(x) → hidden_x
#           - Compute β(z) → hidden_z
#           - Compute ψ(hidden_x) → y_hat
#         Returns:
#             y_hat (Tensor [batch]), hidden_x (Tensor [batch, hidden]), hidden_z (Tensor [batch, hidden])
#         """
#         hidden_x = self.phi(x)
#         hidden_z = self.beta(z)
#         y_hat = self.classifier(hidden_x).view(-1)
#         return y_hat, hidden_x, hidden_z

#     def train_model(self,
#                     X: torch.Tensor,
#                     Z: torch.Tensor,
#                     y: torch.Tensor,
#                     x_val: torch.Tensor = None,
#                     y_val: torch.Tensor = None,
#                     z_val: torch.Tensor = None,
#                     record_loss: bool = False,
#                     early_stopping_patience: int = 10):
#         """
#         Train φ + β + ψ with view alignment loss.

#         Args:
#             X (Tensor): View1 training features.
#             Z (Tensor): View2 training features.
#             y (Tensor): Training labels.
#             x_val, y_val, z_val (Tensor): Optional validation data for early stopping.
#             record_loss (bool): If True, returns (train_losses, val_losses).
#             early_stopping_patience (int): Patience for early stopping on validation BCE loss.

#         Returns:
#             (train_losses, val_losses) if record_loss else None
#         """
#         train_losses, val_losses = [], []
#         best_val_loss = float('inf')
#         patience_counter = 0

#         for _ in range(self.epochs):
#             self.train()
#             self.opt.zero_grad()

#             y_pred, hidden_x, hidden_z = self.forward(X, Z)
#             loss_main = self.bce(y_pred, y.view(-1))
#             loss_align = self.mse(hidden_x, hidden_z)
#             loss = loss_main + self.lambda_view * loss_align

#             loss.backward()
#             self.opt.step()

#             if record_loss:
#                 train_losses.append(loss_main.item())

#             # Early stopping using validation BCE only
#             if x_val is not None and y_val is not None and z_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     y_val_pred, _, _ = self.forward(x_val, z_val)
#                     val_loss_main = self.bce(y_val_pred, y_val.view(-1))
#                 if record_loss:
#                     val_losses.append(val_loss_main.item())

#                 if val_loss_main.item() < best_val_loss:
#                     best_val_loss = val_loss_main.item()
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#                     if patience_counter >= early_stopping_patience:
#                         break

#         if record_loss:
#             return train_losses, val_losses

#     def evaluate(self, X: torch.Tensor, y: torch.Tensor, Z: torch.Tensor = None):
#         """
#         Evaluate model on test set: compute AUC and per-class P/R/F.

#         Args:
#             X (Tensor): View1 test features.
#             y (Tensor): Test labels.
#             Z (Tensor): View2 test features (zeros for test).

#         Returns:
#             Dict with keys: "auc", "p0", "r0", "f0", "p1", "r1", "f1"
#         """
#         self.eval()
#         with torch.no_grad():
#             if Z is None:
#                 # If Z is None, feed a zero tensor of matching shape
#                 Z_dummy = torch.zeros_like(X)
#                 y_pred, _, _ = self.forward(X, Z_dummy)
#             else:
#                 y_pred, _, _ = self.forward(X, Z)

#         y_true = y.view(-1).cpu().numpy()
#         y_score = y_pred.cpu().numpy()
#         y_hat = (y_score >= 0.5).astype(float)

#         p, r, f, _ = precision_recall_fscore_support(y_true, y_hat, zero_division=0)
#         try:
#             auc = roc_auc_score(y_true, y_score)
#         except ValueError:
#             auc = 0.0

#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#         }

# def main():
#     """
#     Main entry point to run Multi‐View experiments for:
#       1) Two‐Loss pretrain + finetune model (MultiViewNN_TwoLoss)
#       2) Simultaneous two‐loss model (MultiViewNN_Simul)

#     Steps:
#       - Set random seed for reproducibility.
#       - Create (or overwrite) results CSV with header.
#       - Load and preprocess data into (grid, train, test) splits.
#       - For each model:
#           * Run num_runs disjoint experiments with internal 3-fold CV.
#           * Aggregate per-run metrics and write to CSV.
#           * Print final aggregated results.
#     """
#     # 1) Ensure reproducibility
#     set_seed(42)

#     # 2) Prepare output CSV
#     OUTPUT_CSV = "multiview_disjoint_results.csv"
#     if os.path.exists(OUTPUT_CSV):
#         os.remove(OUTPUT_CSV)
#     # Header: Method, AUC, p0, r0, f0, p1, r1, f1, Params
#     with open(OUTPUT_CSV, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "Method", "AUC", "p0", "r0", "f0", "p1", "r1", "f1", "Best_Params_per_run"
#         ])

#     # 3) Load & preprocess data
#     dp = MultiViewDatasetPreprocessor(
#         side_info_path="prompting/augmented_data_70B.csv",
#         dataset_id=891
#     )
#     (_, _, _), (train_X, train_y, train_Z), (test_X, test_y, test_Z) = dp.preprocess()

#     # 4) Define experiment configurations
#     experiments = [
#         # {
#         #     "name": "TwoLoss",
#         #     "model": MultiViewNN_TwoLoss,
#         #     "param_grid": {
#         #         "hidden": [64, 128, 256],
#         #         "num_layers_x": [1, 2, 3, 4],
#         #         "num_layers_z": [1, 2, 3, 4],
#         #         "lr": [0.001, 0.01],
#         #         "epochs": [100],
#         #         "lambda_aux": [0.3, 0.5]
#         #     }
#         # },
#         {
#             "name": "Simul",
#             "model": MultiViewNN_Simul,
#             "param_grid": {
#                 "hidden": [64, 128, 256],
#                 "num_layers_x": [1, 2, 3, 4],
#                 "num_layers_z": [1, 2, 3, 4],
#                 "lr": [0.01, 0.001],
#                 "epochs": [100],
#                 "lambda_aux": [0.3, 0.5]
#             }
#         },


#         # # {
#         # #     "name": "MT-MLP-Simul",
#         # #     "model": SimultaneousMultiViewMultiTaskNN,
#         # #     "param_grid": {
#         # #         "hidden": [64],
#         # #         "lr": [0.001, 0.01],
#         # #         "epochs": [100],
#         # #         "lambda_aux": [0.3, 0.5],
#         # #         "lambda_direct": [0.3, 0.5]
#         # #     }
#         # # },
#         # {
#         #     "name": "MV-MLP",
#         #     "model": MultiViewMLP,
#         #     "param_grid": {
#         #         "hidden": [32, 64],
#         #         "num_layers_x": [2, 3],
#         #         "num_layers_z": [1, 2],
#         #         "lr": [0.001, 0.01],
#         #         "lambda_view": [0.3, 0.5],
#         #         "epochs": [100]
#         #     }
#         # }
#     ]

#     # 5) Loop over each experiment
#     for cfg in experiments:
#         name        = cfg["name"]
#         ModelClass  = cfg["model"]
#         param_grid  = cfg["param_grid"]

#         print(f"\n=== Running {name} for {cfg['model'].__name__} ===")

#         # === FIXED CALL SIGNATURE ===
#         metrics_list, best_params_list = run_experiments_multiview(
#             model_class   = ModelClass,
#             method_name   = name,
#             num_runs      = 10,
#             param_grid    = param_grid,
#             train_X_pool  = train_X,    # <— matches function signature
#             train_Z_pool  = train_Z,
#             train_y_pool  = train_y,
#             test_X_pool   = test_X,
#             test_Z_pool   = test_Z,
#             test_y_pool   = test_y,
#             cv            = 3
#         )

#         # Aggregate metrics across runs (mean ± std for each metric)
#         aggregated = aggregate_metrics(metrics_list)

#         # === FIXED CSV WRITING (use 'aggregated', not undefined 'agg') ===
#         write_results_csv(
#             OUTPUT_CSV,
#             method = name,
#             metrics = aggregated,
#             params = str(best_params_list)
#         )

#         # Print final aggregated results to console
#         print_aggregated_results(name, aggregated, str(best_params_list))


# if __name__ == "__main__":
#     main()