# baseline_project.py

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

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ucimlrepo import fetch_ucirepo


# =========================
#   GLOBAL SEED FUNCTION
# =========================

def set_seed(seed: int):
    """
    Set the random seed for reproducibility across random, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
#   DATA PREPROCESSOR
# =========================

class BaselineDatasetPreprocessor:
    """
    Fetches a UCI dataset by ID and performs preprocessing:
      1. Drop 'ID' column if it exists.
      2. Perform a stratified train-test split.
      3. Standardize the 'BMI' column using z-score.
      4. Undersample the training set to balance classes (up to 375 samples per class).
      5. Return three sets:
         - grid_x, grid_y: balanced training set (undersampled)
         - train_pool_x, train_pool_y: remaining training set
         - test_x, test_y: test set (with standardized 'BMI' and labels attached)
    """

    def __init__(self, dataset_id: int = 891):
        """
        Args:
            dataset_id (int): UCI dataset ID. Default is 891.
        """
        # Fetch from UCI ML Repository
        self.original = fetch_ucirepo(id=dataset_id)
        self.X = self.original.data.features  # pandas DataFrame of features
        self.y = self.original.data.targets   # pandas Series of target labels
        self.scaler = StandardScaler()        # StandardScaler for 'BMI' column

    def preprocess(self):
        """
        Execute preprocessing and return tensors for:
          - balanced grid training set
          - train pool (remaining training examples)
          - test set
        
        Returns:
            tuple: (grid_x, grid_y, train_pool_x, train_pool_y, test_x, test_y)
        """
        # 1. Drop 'ID' column if present
        if 'ID' in self.X.columns:
            df = self.X.drop(columns=['ID']).copy()
        else:
            df = self.X.copy()

        # 2. Stratified train-test split (25% test)
        train_x, test_x, train_y, test_y = train_test_split(
            df,
            self.y,
            test_size=0.25,
            random_state=42,
            stratify=self.y
        )

        # 3. Standardize 'BMI' in training set
        train_x['BMI'] = self.scaler.fit_transform(train_x[['BMI']])
        train_x['label'] = train_y.values  # attach labels to training DataFrame

        # 4. Balance training set by undersampling to 375 samples per class
        pos = train_x[train_x['label'] == 1]
        neg = train_x[train_x['label'] == 0]
        n = min(len(pos), len(neg), 375)
        grid = pd.concat([
            pos.sample(n, random_state=42),
            neg.sample(n, random_state=42)
        ])
        # Remaining examples form the train pool
        train_pool = train_x.drop(grid.index)

        # 5. Standardize 'BMI' in test set (use same scaler)
        test_x['BMI'] = self.scaler.transform(test_x[['BMI']])
        test_pool = test_x.copy()
        test_pool['label'] = test_y.values  # attach labels to test DataFrame

        # Helper: convert pandas DataFrame to PyTorch tensors
        def to_tensor(df: pd.DataFrame):
            x = torch.tensor(df.drop(columns=['label']).values, dtype=torch.float32)
            y = torch.tensor(df['label'].values, dtype=torch.float32)
            return x, y

        # Convert each set to tensors
        grid_x, grid_y = to_tensor(grid)
        pool_x, pool_y = to_tensor(train_pool)
        test_x_tensor, test_y_tensor = to_tensor(test_pool)

        return grid_x, grid_y, pool_x, pool_y, test_x_tensor, test_y_tensor


# =========================
#       MODEL CLASSES
# =========================

SEED = 42  # Global seed for model randomness

class BaselineMLP(nn.Module):
    """
    Simple multi-layer perceptron for binary classification:
      - Configurable number of hidden layers and hidden dimensions.
      - Uses BCE loss and Adam optimizer.
      - Sigmoid output for probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, num_layers: int = 1,
                 lr: float = 0.01, epochs: int = 300):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of units in each hidden layer. Default 16.
            num_layers (int): Number of hidden layers. Default 1.
            lr (float): Learning rate for Adam optimizer. Default 0.01.
            epochs (int): Maximum number of training epochs. Default 300.
        """
        super().__init__()
        self.epochs = epochs

        # Build hidden layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU()
            )
            for i in range(num_layers)
        ])

        # Output layer: single unit + Sigmoid for probability
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all hidden layers and output.
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.output(h).view(-1)

    def train_model(self, x: torch.Tensor, y: torch.Tensor,
                    x_val: torch.Tensor = None, y_val: torch.Tensor = None,
                    early_stopping: int = 10,
                    record_loss: bool = False, **kwargs):
        """
        Train the MLP model.
        Args:
            x (Tensor): Training features.
            y (Tensor): Training labels (0 or 1).
            x_val (Tensor, optional): Validation features.
            y_val (Tensor, optional): Validation labels.
            early_stopping (int): Number of epochs with no improvement to stop early.
            record_loss (bool): Whether to record loss history for plotting.
        Returns:
            tuple: (train_loss_history, val_loss_history)
        """
        train_hist = []
        val_hist = []
        best_val = float('inf')
        wait = 0

        for epoch in range(self.epochs):
            # Training step
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.loss_fn(out, y.view(-1))
            loss.backward()
            self.optimizer.step()

            if record_loss:
                train_hist.append(loss.item())

            # Validation step (if provided)
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_out = self.forward(x_val)
                    val_loss = self.loss_fn(val_out, y_val.view(-1))
                if record_loss:
                    val_hist.append(val_loss.item())

                # Early stopping check
                if val_loss.item() < best_val:
                    best_val = val_loss.item()
                    wait = 0
                else:
                    wait += 1
                    if wait >= early_stopping:
                        break

        return train_hist, val_hist

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Evaluate the MLP on a given dataset.
        Returns a dictionary of metrics: AUC, accuracy, class-wise precision/recall/F1, BCE loss.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)

        # Convert probabilities to binary predictions (0 or 1)
        preds = (out >= 0.5).float().cpu().numpy()
        y_np = y.view(-1).cpu().numpy()

        # Compute metrics
        auc = roc_auc_score(y_np, out.cpu().numpy())
        p, r, f, _ = precision_recall_fscore_support(y_np, preds, zero_division=0)
        overall_acc = np.mean(preds == y_np)
        a0 = np.mean(preds[y_np == 0] == 0) if np.sum(y_np == 0) > 0 else 0.0
        a1 = np.mean(preds[y_np == 1] == 1) if np.sum(y_np == 1) > 0 else 0.0
        loss = self.loss_fn(out, y.view(-1)).item()

        return {
            'auc': auc,
            'overall_acc': overall_acc,
            'a0': a0, 'p0': p[0], 'r0': r[0], 'f0': f[0],
            'a1': a1, 'p1': p[1], 'r1': r[1], 'f1': f[1],
            'bce_loss': loss
        }


class BaselineLogisticRegression(nn.Module):
    """
    Simple logistic regression implemented in PyTorch:
      - Single linear layer + Sigmoid.
      - Uses BCE loss and Adam optimizer.
    """

    def __init__(self, input_dim: int, lr: float = 0.01, epochs: int = 300):
        """
        Args:
            input_dim (int): Number of input features.
            lr (float): Learning rate for Adam optimizer. Default 0.01.
            epochs (int): Maximum number of training epochs. Default 300.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()
        self.epochs = epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the linear layer and sigmoid.
        """
        return self.model(x).view(-1)

    def train_model(self, x: torch.Tensor, y: torch.Tensor,
                    x_val: torch.Tensor = None, y_val: torch.Tensor = None,
                    early_stopping: int = 10,
                    record_loss: bool = False, **kwargs):
        """
        Train the logistic regression model.
        Args:
            x (Tensor): Training features.
            y (Tensor): Training labels.
            x_val (Tensor, optional): Validation features.
            y_val (Tensor, optional): Validation labels.
            early_stopping (int): Number of epochs with no improvement to stop early.
            record_loss (bool): Whether to record loss history.
        Returns:
            tuple: (train_loss_history, val_loss_history)
        """
        train_hist = []
        val_hist = []
        best_val = float('inf')
        wait = 0

        for epoch in range(self.epochs):
            # Training step
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.loss_fn(out, y.view(-1))
            loss.backward()
            self.optimizer.step()

            if record_loss:
                train_hist.append(loss.item())

            # Validation step
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_out = self.forward(x_val)
                    vloss = self.loss_fn(val_out, y_val.view(-1))
                if record_loss:
                    val_hist.append(vloss.item())

                # Early stopping check
                if vloss.item() < best_val:
                    best_val = vloss.item()
                    wait = 0
                else:
                    wait += 1
                    if wait >= early_stopping:
                        break

        return train_hist, val_hist

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Evaluate logistic regression model on given dataset.
        Returns metrics: AUC, accuracy, class-wise precision/recall/F1, log loss.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x).cpu().numpy()

        preds = (out >= 0.5).astype(float)
        y_np = y.cpu().numpy()

        # Compute metrics
        auc = roc_auc_score(y_np, out)
        p, r, f, _ = precision_recall_fscore_support(y_np, preds, zero_division=0)
        overall_acc = np.mean(preds == y_np)
        a0 = np.mean(preds[y_np == 0] == 0) if np.sum(y_np == 0) > 0 else 0.0
        a1 = np.mean(preds[y_np == 1] == 1) if np.sum(y_np == 1) > 0 else 0.0
        bce = log_loss(y_np, out)

        return {
            'auc': auc,
            'overall_acc': overall_acc,
            'a0': a0, 'p0': p[0], 'r0': r[0], 'f0': f[0],
            'a1': a1, 'p1': p[1], 'r1': r[1], 'f1': f[1],
            'bce_loss': bce
        }


class XGBoostModel:
    """
    Wrapper for XGBoostClassifier:
      - Accepts numpy arrays or PyTorch tensors.
      - Performs optional early stopping if validation is provided.
    """

    def __init__(self, learning_rate: float = 0.01, n_estimators: int = 300,
                 max_depth: int = 3, early_stopping: int = 10):
        """
        Args:
            learning_rate (float): Learning rate (eta) for XGBoost. Default 0.01.
            n_estimators (int): Maximum number of boosting rounds. Default 300.
            max_depth (int): Maximum tree depth. Default 3.
            early_stopping (int): Early stopping rounds. Default 10.
        """
        self.model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=SEED
        )
        self.early_rounds = early_stopping

    def train_model(self, X_tr, y_tr,
                    x_val=None, y_val=None,
                    early_stopping: int = 10,
                    record_loss: bool = False, **kwargs):
        """
        Train XGBoost model.
        Args:
            X_tr (array or Tensor): Training features.
            y_tr (array or Tensor): Training labels.
            x_val (array or Tensor, optional): Validation features.
            y_val (array or Tensor, optional): Validation labels.
            early_stopping (int): Early stopping rounds. Default 10.
            record_loss (bool): Whether to record train/val logloss history.
        Returns:
            tuple: (train_loss_history, val_loss_history)
        """
        # Convert PyTorch tensors to numpy if needed
        Xn = X_tr.numpy() if hasattr(X_tr, 'numpy') else X_tr
        yn = y_tr.numpy() if hasattr(y_tr, 'numpy') else y_tr

        train_hist = []
        val_hist = []

        if x_val is not None and y_val is not None:
            Xv = x_val.numpy() if hasattr(x_val, 'numpy') else x_val
            yv = y_val.numpy() if hasattr(y_val, 'numpy') else y_val

            if record_loss:
                # Fit with eval_set to record training and validation logloss
                self.model.fit(
                    Xn, yn,
                    eval_set=[(Xn, yn), (Xv, yv)],
                    early_stopping_rounds=self.early_rounds,
                    verbose=False
                )
                evals = self.model.evals_result()
                train_hist = evals['validation_0']['logloss']
                val_hist = evals['validation_1']['logloss']
            else:
                # Fit with only validation for early stopping
                self.model.fit(
                    Xn, yn,
                    eval_set=[(Xv, yv)],
                    early_stopping_rounds=self.early_rounds,
                    verbose=False
                )
        else:
            # No validation provided: fit on all training data
            self.model.fit(Xn, yn, verbose=False)

        return train_hist, val_hist

    def evaluate(self, X_te, y_te) -> dict:
        """
        Evaluate XGBoost model on test data.
        Returns metrics: AUC, class-wise metrics, log loss.
        """
        Xn = X_te.numpy() if hasattr(X_te, 'numpy') else X_te
        yn = y_te.numpy() if hasattr(y_te, 'numpy') else y_te

        # Predicted probabilities for class 1
        proba = self.model.predict_proba(Xn)[:, 1]
        preds = (proba >= 0.5).astype(float)

        # Compute metrics
        auc = roc_auc_score(yn, proba)
        p, r, f, _ = precision_recall_fscore_support(yn, preds, zero_division=0)
        overall_acc = np.mean(preds == yn)
        a0 = np.mean(preds[yn == 0] == 0) if np.sum(yn == 0) > 0 else 0.0
        a1 = np.mean(preds[yn == 1] == 1) if np.sum(yn == 1) > 0 else 0.0
        bce = log_loss(yn, proba)

        return {
            'auc': auc,
            'overall_acc': overall_acc,
            'a0': a0, 'p0': p[0], 'r0': r[0], 'f0': f[0],
            'a1': a1, 'p1': p[1], 'r1': r[1], 'f1': f[1],
            'bce_loss': bce
        }


class RandomForestModel:
    """
    Wrapper for scikit-learn's RandomForestClassifier:
      - Trains on numpy arrays or PyTorch tensors.
      - Optionally records training and validation log loss.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        """
        Args:
            n_estimators (int): Number of trees in the forest. Default 100.
            max_depth (int): Maximum tree depth. Default None (no limit).
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=SEED
        )

    def train_model(self, X_tr, y_tr,
                    x_val=None, y_val=None,
                    early_stopping: int = 10,
                    record_loss: bool = False, **kwargs):
        """
        Train RandomForest model.
        Args:
            X_tr (array or Tensor): Training features.
            y_tr (array or Tensor): Training labels.
            x_val (array or Tensor, optional): Validation features.
            y_val (array or Tensor, optional): Validation labels.
            record_loss (bool): Whether to record training/validation log loss.
        Returns:
            tuple: (train_loss_list, val_loss_list)
        """
        # Convert to numpy arrays if given as tensors
        Xn = X_tr.numpy() if hasattr(X_tr, 'numpy') else X_tr
        yn = y_tr.numpy() if hasattr(y_tr, 'numpy') else y_tr

        # Fit the RandomForest
        self.model.fit(Xn, yn)

        train_hist = []
        val_hist = []

        if record_loss:
            # Compute training log loss
            train_proba = self.model.predict_proba(Xn)[:, 1]
            train_loss = log_loss(yn, train_proba)
            train_hist = [train_loss]

            # Compute validation log loss if provided
            if x_val is not None and y_val is not None:
                Xv = x_val.numpy() if hasattr(x_val, 'numpy') else x_val
                yv = y_val.numpy() if hasattr(y_val, 'numpy') else y_val
                val_proba = self.model.predict_proba(Xv)[:, 1]
                val_loss = log_loss(yv, val_proba)
                val_hist = [val_loss]

        return train_hist, val_hist

    def evaluate(self, X_te, y_te) -> dict:
        """
        Evaluate RandomForest on test data.
        Returns metrics: AUC, class-wise metrics, log loss.
        """
        Xn = X_te.numpy() if hasattr(X_te, 'numpy') else X_te
        yn = y_te.numpy() if hasattr(y_te, 'numpy') else y_te

        proba = self.model.predict_proba(Xn)[:, 1]
        preds = (proba >= 0.5).astype(int)

        auc = roc_auc_score(yn, proba)
        p, r, f, _ = precision_recall_fscore_support(yn, preds, zero_division=0)
        bce = log_loss(yn, proba)
        acc0 = np.mean(preds[yn == 0] == 0) if (yn == 0).any() else 0.0
        acc1 = np.mean(preds[yn == 1] == 1) if (yn == 1).any() else 0.0

        return {
            'auc': auc,
            'a0': acc0, 'p0': p[0], 'r0': r[0], 'f0': f[0],
            'a1': acc1, 'p1': p[1], 'r1': r[1], 'f1': f[1],
            'bce_loss': bce
        }


# =========================
# run_experiments_baseline
# =========================
def run_experiments_baseline(
    model_class,
    method_name: str,
    num_runs: int,
    param_grid: dict,
    train_x_pool: torch.Tensor,
    train_y_pool: torch.Tensor,
    test_x_pool: torch.Tensor,
    test_y_pool: torch.Tensor,
    cv: int = 3
) -> list:
    """
    Combined function that, for each iteration:
      1. Samples 250 positives + 250 negatives *without replacement* from train_pool → 500 train samples.
      2. Samples 125 test examples (may overlap across iterations) from test_pool.
      3. Performs 3-fold CV on those 500 train samples to find best hyperparameters.
      4. Retrains model on all 500 train samples using best hyperparameters.
      5. Evaluates the retrained model on the 125-sample test set → collects metrics.
      6. Repeats num_runs times, ensuring that each 500-sample training set does not overlap across iterations.

    Returns:
        metrics_list (List[dict]): One dict of evaluation metrics per iteration.
        best_params_list (List[dict]): The best_params dict found in each iteration.
    """
    import os
    import random
    import itertools
    import numpy as np
    import torch
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
    import inspect

    metrics_list = []
    best_params_list = []

    # Separate positive/negative indices from the train pool
    train_df = pd.DataFrame(train_x_pool.numpy())
    train_df['Diabetes_binary'] = train_y_pool.numpy()
    class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
    class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
    test_pool_size = len(test_x_pool)

    # random.shuffle(pos_indices)
    # random.shuffle(neg_indices)

    # Pointers to move through positive/negative pools
    # pos_ptr = 0
    # neg_ptr = 0

    # Get the signature of the model __init__ to know which params to pass
    sig = inspect.signature(model_class.__init__).parameters

    # Ensure 'img/' directory exists for plots
    os.makedirs("img", exist_ok=True)

    for run_idx in range(1, num_runs + 1):
        # ----------------------------
        # 1) Balanced sampling: up to 250 positives and negatives
        # ----------------------------
        sampled1 = random.sample(class1_indices, min(250, len(class1_indices)))
        sampled0 = random.sample(class0_indices, min(250, len(class0_indices)))
        train_idx = sampled1 + sampled0
        x_train = train_x_pool[train_idx]
        y_train = train_y_pool[train_idx]

        # --------------------------------------------
        # 2) Sample random subset of test pool (125 examples)
        # --------------------------------------------
        test_idx = random.sample(range(test_pool_size), min(125, test_pool_size))
        x_test = test_x_pool[test_idx]
        y_test = test_y_pool[test_idx]

        # --------------------------------------------
        # 3) 3-fold CV를 통한 그리드 서치: 최적 하이퍼파라미터 탐색
        # --------------------------------------------
        best_auc = -1.0
        best_params = None
        keys = list(param_grid.keys())

        # 3-fold split (sklearn KFold)
        X_np = x_train.numpy()
        y_np = y_train.numpy()
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        # Grid Search
        for values in itertools.product(*[param_grid[k] for k in keys]):
            params = dict(zip(keys, values))
            fold_aucs = []

            # For each forld - train/val split
            for tr_idx, val_idx in kf.split(X_np):
                x_tr = torch.tensor(X_np[tr_idx], dtype=torch.float32)
                y_tr = torch.tensor(y_np[tr_idx], dtype=torch.float32)
                x_val = torch.tensor(X_np[val_idx], dtype=torch.float32)
                y_val = torch.tensor(y_np[val_idx], dtype=torch.float32)

                # create model instance 
                init_kwargs = {}
                if 'input_dim' in sig:
                    init_kwargs['input_dim'] = X_np.shape[1]
                for k, v in params.items():
                    if k in sig:
                        init_kwargs[k] = v
                model = model_class(**init_kwargs)

                model.train_model(
                    x_tr, y_tr,
                    x_val=x_val, y_val=y_val,
                    early_stopping=10
                )

                # Compute validation AUC for this fold
                with torch.no_grad():
                    if hasattr(model, 'forward'):
                        out_val = model.forward(x_val)
                    else:
                        proba = model.model.predict_proba(x_val.numpy())[:, 1]
                        out_val = torch.from_numpy(proba).float()

                    auc_score = roc_auc_score(y_val.cpu().numpy(), out_val.cpu().numpy())
                    fold_aucs.append(auc_score)

            avg_auc = float(np.mean(fold_aucs))
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_params = params.copy()

        print(f"[{method_name}] Iteration {run_idx} → Best CV params: {best_params}, Avg CV AUC: {best_auc:.4f}")
        best_params_list.append(best_params)

        # --------------------------------------------
        # 4) Retrain on all 500 samples with best_params
        # --------------------------------------------
        init_kwargs = {}
        if 'input_dim' in sig:
            init_kwargs['input_dim'] = x_train.shape[1]
        for k, v in best_params.items():
            if k in sig:
                init_kwargs[k] = v
        best_model = model_class(**init_kwargs)
        best_model.train_model(x_train, y_train, early_stopping=10)

        # --------------------------------------------
        # 5) Evaluate on the 125-sample test set
        # --------------------------------------------
        metrics = best_model.evaluate(x_test, y_test)
        print(f"[{method_name}] Iteration {run_idx} → Test metrics: {metrics}")
        metrics_list.append(metrics)

    return metrics_list, best_params_list

        # (Optional) Loss curve를 기록했다면 아래처럼 저장할 수 있음
        # plt.figure()
        # plt.plot(train_loss_history), plt.plot(val_loss_history)
        # plt.savefig(f"img/{method_name}_run{run_idx}.png")

    # return metrics_list

def aggregate_metrics(metrics_list: list) -> dict:
    """
    Aggregate a list of metric dictionaries by computing mean ± std for each metric key.
    
    Args:
        metrics_list (list): List of dicts, each containing the same set of metric keys.
    
    Returns:
        dict: Dictionary where each value is formatted as 'mean ± std'.
    """
    agg = {}
    for key in metrics_list[0].keys():
        vals = [m[key] for m in metrics_list]
        agg[key] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
    return agg


def write_results_csv(
    filename: str,
    method: str,
    metrics: dict,
    params: dict
):
    """
    Append a row of results (metrics + params) to a CSV file.
    
    Args:
        filename (str): CSV filename (will be created/appended).
        method (str): Name of the method/model.
        metrics (dict): Metric dictionary (keys like 'auc', 'overall_acc', etc.).
        params (dict): Hyperparameter dict used for this result.
    """
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            method,
            metrics.get('auc', ''),
            metrics.get('overall_acc', ''),
            metrics.get('a0', ''), metrics.get('p0', ''), metrics.get('r0', ''), metrics.get('f0', ''),
            metrics.get('a1', ''), metrics.get('p1', ''), metrics.get('r1', ''), metrics.get('f1', ''),
            metrics.get('bce_loss', ''),
            params
        ])


def print_aggregated_results(
    method: str,
    metrics: dict,
    params: str
):
    """
    Print aggregated results in a readable format.
    
    Args:
        method (str): Name of the method/model.
        metrics (dict): Aggregated metric dictionary (formatted strings).
        params (str): String representation of best hyperparameters.
    """
    print("\n=== Final Aggregated Results ===")
    print(f"Method: {method}")
    print(f"AUC: {metrics.get('auc','')}")
    print(f"Overall Accuracy: {metrics.get('overall_acc','')}")
    print(f"Class0 -> Accuracy: {metrics.get('a0','')}, Precision: {metrics.get('p0','')}, Recall: {metrics.get('r0','')}, F1: {metrics.get('f0','')}")
    print(f"Class1 -> Accuracy: {metrics.get('a1','')}, Precision: {metrics.get('p1','')}, Recall: {metrics.get('r1','')}, F1: {metrics.get('f1','')}")
    print(f"Test BCE Loss: {metrics.get('bce_loss','')}")
    print(f"Best Parameter Set: {params}")
    print("---------------------------------------------------")


# =========================
#          MAIN
# =========================

def main():
    """
    Main function to run the entire baseline project:
      1. Set seed for reproducibility.
      2. Load and preprocess data via BaselineDatasetPreprocessor.
      3. For each model (MLP, LogisticRegression, XGBoost, RandomForest):
         a. Perform grid search CV on the balanced 'grid' set.
         b. Report best_params and best AUC.
         c. Run repeated experiments on train/test pools to get final metrics.
         d. Aggregate metrics and print/save results.
    """
    # 1. Ensure reproducibility
    set_seed(SEED)

    # 2. Data loading and preprocessing
    dp = BaselineDatasetPreprocessor(dataset_id=891)
    grid_x, grid_y, pool_x, pool_y, test_x, test_y = dp.preprocess()

    # 3. Define experiments for each model
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

    # CSV file to record all final results
    results_csv = "baseline_results.csv"
    # Write header if file does not exist
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Method", "AUC", "Overall_Acc",
                "A0", "P0", "R0", "F0",
                "A1", "P1", "R1", "F1",
                "BCE_Loss", "Params"
            ])

    # Loop over each experiment configuration
    for cfg in experiments:
        name = cfg["name"]
        ModelClass = cfg["model"]
        param_grid = cfg["param_grid"]

        # If GPU is available, clear cache (not strictly necessary)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # # a. Hyperparameter grid search on the balanced grid set
        # best_params, best_auc = grid_search_cv_basline(
        #     ModelClass,
        #     param_grid,
        #     grid_x,
        #     grid_y,
        #     cv=3
        # )
        # print(f"[{name}] Best params: {best_params}, Best CV AUC: {best_auc:.4f}")

        # b. Run repeated experiments on train/test pools and collect metrics
        metrics_list, best_params_list = run_experiments_baseline(
            model_class=ModelClass,
            method_name=name,
            num_runs=10,
            param_grid=param_grid,
            train_x_pool=pool_x,
            train_y_pool=pool_y,
            test_x_pool=test_x,
            test_y_pool=test_y,
            cv=3
        )

        # c. Print each iteration’s best_params
        print(f"\n[{name}] Best hyperparameters per iteration:")
        for i, bp in enumerate(best_params_list, start=1):
            print(f"  Iteration {i}: {bp}")

        # d. Compute overall mean±std for each metric across the 10 iterations
        #    Example: metrics_list is a list of dicts, each dict has keys 'auc', 'overall_acc', etc.
        aggregated = {}
        for metric_key in metrics_list[0].keys():
            values = [m[metric_key] for m in metrics_list]
            aggregated[metric_key] = (np.mean(values), np.std(values))

        # Print aggregated results
        print(f"\n=== [{name}] Final Aggregated Metrics (mean ± std) ===")
        for metric_key, (mean_val, std_val) in aggregated.items():
            print(f"  {metric_key}: {mean_val:.4f} ± {std_val:.4f}")

        # # c. Aggregate metrics (mean ± std)
        # aggregated = aggregate_metrics(metrics_list)

        # # d. Print aggregated results and append to CSV
        # print_aggregated_results(name, aggregated, str(best_params))
        # write_results_csv(results_csv, name, aggregated, best_params)

if __name__ == "__main__":
    main()