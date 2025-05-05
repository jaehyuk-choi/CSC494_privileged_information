# model/baseline_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression as SklearnLR
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

SEED = 42

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=1, lr=0.01, epochs=300):
        super().__init__()
        self.epochs = epochs
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU()
            ) for i in range(num_layers)
        ])
        self.output = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        h = x
        for l in self.layers:
            h = l(h)
        return self.output(h).view(-1)

    def train_model(self, x, y,
                    x_val=None, y_val=None,
                    early_stopping=10,
                    record_loss=False, **kwargs):
        train_hist, val_hist = [], []
        best_val, wait = float('inf'), 0

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.loss_fn(out, y.view(-1))
            loss.backward()
            self.optimizer.step()

            if record_loss:
                train_hist.append(loss.item())

            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    v = self.forward(x_val)
                    l2 = self.loss_fn(v, y_val.view(-1))
                if record_loss:
                    val_hist.append(l2.item())
                if l2.item() < best_val:
                    best_val, wait = l2.item(), 0
                else:
                    wait += 1
                    if wait >= early_stopping:
                        break

        return train_hist, val_hist

    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        preds = (out >= 0.5).float().cpu().numpy()
        y_np = y.view(-1).cpu().numpy()
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
    def __init__(self, input_dim, lr=0.01, epochs=300):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()
        self.epochs = epochs

    def forward(self, x):
        return self.model(x).view(-1)

    def train_model(self, x, y,
                    x_val=None, y_val=None,
                    early_stopping=10,
                    record_loss=False, **kwargs):
        train_hist, val_hist = [], []
        best_val, wait = float('inf'), 0

        for _ in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            out = self.forward(x)
            loss = self.loss_fn(out, y.view(-1))
            loss.backward()
            self.optimizer.step()

            if record_loss:
                train_hist.append(loss.item())

            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_out = self.forward(x_val)
                    vloss = self.loss_fn(val_out, y_val.view(-1))
                if record_loss:
                    val_hist.append(vloss.item())
                if vloss.item() < best_val:
                    best_val, wait = vloss.item(), 0
                else:
                    wait += 1
                    if wait >= early_stopping:
                        break

        return train_hist, val_hist

    def evaluate(self, x, y):
        self.eval()
        out = self.forward(x).detach().cpu().numpy()
        preds = (out >= 0.5).astype(float)
        y_np = y.cpu().numpy()
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
    def __init__(self, learning_rate=0.01, n_estimators=300, max_depth=3, early_stopping=10):
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
                    early_stopping=10,
                    record_loss=False, **kwargs):
        Xn = X_tr.numpy() if hasattr(X_tr, 'numpy') else X_tr
        yn = y_tr.numpy() if hasattr(y_tr, 'numpy') else y_tr
        train_hist, val_hist = [], []
        if x_val is not None and y_val is not None:
            Xv = x_val.numpy() if hasattr(x_val, 'numpy') else x_val
            yv = y_val.numpy() if hasattr(y_val, 'numpy') else y_val
            if record_loss:
                self.model.fit(
                    Xn, yn,
                    eval_set=[(Xn, yn), (Xv, yv)],
                    early_stopping_rounds=self.early_rounds,
                    verbose=False
                )
                evals = self.model.evals_result()
                train_hist = evals['validation_0']['logloss']
                val_hist   = evals['validation_1']['logloss']
            else:
                self.model.fit(
                    Xn, yn,
                    eval_set=[(Xv, yv)],
                    early_stopping_rounds=self.early_rounds,
                    verbose=False
                )
        else:
            self.model.fit(Xn, yn, verbose=False)
        return train_hist, val_hist

    def evaluate(self, X_te, y_te):
        Xn = X_te.numpy() if hasattr(X_te, 'numpy') else X_te
        yn = y_te.numpy() if hasattr(y_te, 'numpy') else y_te
        proba = self.model.predict_proba(Xn)[:, 1]
        preds = (proba >= 0.5).astype(float)
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
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=SEED
        )

    def train_model(self, X_tr, y_tr,
                    x_val=None, y_val=None,
                    early_stopping=10,
                    record_loss=False, **kwargs):
        Xn = X_tr.numpy() if hasattr(X_tr, 'numpy') else X_tr
        yn = y_tr.numpy() if hasattr(y_tr, 'numpy') else y_tr
        # Fit the random forest model
        self.model.fit(Xn, yn)
        train_hist, val_hist = [], []
        if record_loss:
            # Calculate training log loss
            train_proba = self.model.predict_proba(Xn)[:, 1]
            train_loss = log_loss(yn, train_proba)
            train_hist = [train_loss]
            # Calculate validation log loss if validation set provided
            if x_val is not None and y_val is not None:
                Xv = x_val.numpy() if hasattr(x_val, 'numpy') else x_val
                yv = y_val.numpy() if hasattr(y_val, 'numpy') else y_val
                val_proba = self.model.predict_proba(Xv)[:, 1]
                val_loss = log_loss(yv, val_proba)
                val_hist = [val_loss]
        return train_hist, val_hist

    def evaluate(self, X_te, y_te):
        Xn = X_te.numpy() if hasattr(X_te, 'numpy') else X_te
        yn = y_te.numpy() if hasattr(y_te, 'numpy') else y_te
        proba = self.model.predict_proba(Xn)[:, 1]
        preds = (proba >= 0.5).astype(int)

        auc = roc_auc_score(yn, proba)
        p, r, f, _ = precision_recall_fscore_support(yn, preds, zero_division=0)
        bce = log_loss(yn, proba)

        acc0 = (preds[yn == 0] == 0).mean() if (yn == 0).any() else 0.0
        acc1 = (preds[yn == 1] == 1).mean() if (yn == 1).any() else 0.0

        return {
            'auc': auc,
            'a0': acc0, 'p0': p[0], 'r0': r[0], 'f0': f[0],
            'a1': acc1, 'p1': p[1], 'r1': r[1], 'f1': f[1],
            'bce_loss': bce
        }
