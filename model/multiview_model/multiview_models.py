# multiview_model/multiview_models.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

################################################################################
# 1) Two‐Loss Multi‐View Model
################################################################################
class MultiViewNN_TwoLoss(nn.Module):
    """
    Two‐branch multi‐view model:
      1) Pretrain auxiliary branch (β+ψ) on Z → y.
      2) Freeze β, copy ψ as teacher.
      3) Train main branch (φ+ψ) on X with auxiliary regularization:
           L = BCE(ψ(φ(x)), y) + λ * BCE(ψ(β(z)), y)
    """
    def __init__(self, dim_x, dim_z, hidden=32, lr=1e-3, epochs=100, lambda_aux=0.5):
        super().__init__()
        self.epochs = epochs
        self.lambda_aux = lambda_aux

        # φ(x): main branch
        self.phi = nn.Sequential(
            nn.Linear(dim_x, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        # β(z): auxiliary branch
        self.beta = nn.Sequential(
            nn.Linear(dim_z, hidden), nn.ReLU()
        )
        # ψ: shared classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

        # optimizers
        self.aux_opt  = optim.Adam(
            list(self.beta.parameters()) + list(self.classifier.parameters()),
            lr=lr
        )
        self.main_opt = None  # set after aux pretraining

        # losses
        self.bce = nn.BCELoss()
        # teacher copy
        self.fixed_classifier = None

    def forward(self, x, z):
        """
        Returns two predictions:
         - y_hat = ψ(φ(x))
         - y_z   = ψ(β(z))
        both as 1‐D tensors of shape (batch,)
        """
        s   = self.phi(x)
        sz  = self.beta(z)
        return self.classifier(s).view(-1), self.classifier(sz).view(-1)

    def pretrain_aux(self, Z, y):
        """Pretrain auxiliary branch on (Z, y)."""
        x_dummy = torch.zeros(Z.shape[0], self.phi[0].in_features, device=Z.device)

        for epoch in range(self.epochs):
            self.aux_opt.zero_grad()
            # 2) after forward, print representation shapes
            y_hat_x, y_hat_z = self.forward(x_dummy, Z)
            
            loss   = self.bce(y_hat_z, y)
            loss.backward()
            self.aux_opt.step()
        # freeze β
        for p in self.beta.parameters():
            p.requires_grad = False
        # copy ψ as teacher
        self.fixed_classifier = copy.deepcopy(self.classifier)
        # create main optimizer for φ+ψ
        self.main_opt = optim.Adam(
            list(self.phi.parameters()) + list(self.classifier.parameters()),
            lr=self.aux_opt.defaults['lr']
        )

    def train_model(self, X, Z, y, x_val = None, y_val = None, z_val=None, record_loss=False, early_stopping_patience=10):
        """Train φ+ψ on (X, y) with auxiliary regularization from Z."""
        train_loss_history, val_loss_history = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for _ in range(self.epochs):
            self.train()
            self.main_opt.zero_grad()
            y_hat, _ = self.forward(X, Z)
            with torch.no_grad():
                t = self.fixed_classifier(self.beta(Z)).view(-1)
            loss = self.bce(y_hat, y) + self.lambda_aux * self.bce(y_hat, t)
            loss.backward()
            self.main_opt.step()
            if record_loss:
                train_loss_history.append(loss.item())

            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    y_val_hat, _ = self.forward(x_val, z_val)
                    val_loss = self.bce(y_val_hat, y_val)
                    if record_loss:
                        val_loss_history.append(val_loss.item())
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            break

        if record_loss:
            return train_loss_history, val_loss_history

    def evaluate(self, X, y, Z):
        """
        Returns metrics dict including:
          - ‘auc’
          - precision/recall/f1 for each class
        """
        self.eval()
        with torch.no_grad():
            y_hat, _ = self.forward(X, Z)
        preds = (y_hat >= 0.5).float().cpu().numpy()
        y_np  = y.cpu().numpy()
        p, r, f, _ = precision_recall_fscore_support(y_np, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_np, y_hat.cpu().numpy())
        except:
            auc = 0.0
        return {"auc":auc,
                "p0":p[0],"r0":r[0],"f0":f[0],
                "p1":p[1],"r1":r[1],"f1":f[1]}

################################################################################
# 2) Simultaneous Two‐Loss Multi‐View Model
################################################################################
class MultiViewNN_Simul(nn.Module):
    """
    Two‐branch simultaneous training:
      L = BCE(ψ(φ(x)), y) + λ * BCE(ψ(β(z)), y)
    trained jointly from scratch.
    """
    def __init__(self, dim_x, dim_z, hidden=32, lr=1e-3, epochs=100, lambda_aux=0.5):
        super().__init__()
        self.epochs     = epochs
        self.lambda_aux = lambda_aux

        self.phi        = nn.Sequential(
            nn.Linear(dim_x, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.beta       = nn.Sequential(
            nn.Linear(dim_z, hidden), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden,1), nn.Sigmoid()
        )
        self.opt        = optim.Adam(self.parameters(), lr=lr)
        self.bce        = nn.BCELoss()

    def forward(self, x, z):
        s   = self.phi(x)
        sz  = self.beta(z)
        return self.classifier(s).view(-1), self.classifier(sz).view(-1)

    def train_model(self, X, Z, y, x_val=None, y_val=None, z_val=None, early_stopping_patience=10, record_loss=False):
        train_loss_history, val_loss_history = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for _ in range(self.epochs):
            self.train()
            self.opt.zero_grad()
            yh, yz = self.forward(X, Z)
            loss = self.bce(yh, y) + self.lambda_aux * self.bce(yz, y)
            loss.backward()
            self.opt.step()
            if record_loss:
                train_loss_history.append(loss.item())

            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    yh_val, _ = self.forward(x_val, z_val)
                    val_loss = self.bce(yh_val, y_val)
                    if record_loss:
                        val_loss_history.append(val_loss.item())
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            break

        if record_loss:
            return train_loss_history, val_loss_history

    def evaluate(self, X, y, Z):
        self.eval()
        with torch.no_grad():
            yh, _ = self.forward(X, Z)
        preds = (yh >= 0.5).float().cpu().numpy()
        y_np  = y.cpu().numpy()
        p, r, f, _ = precision_recall_fscore_support(y_np, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_np, yh.cpu().numpy())
        except:
            auc = 0.0
        return {"auc":auc,
                "p0":p[0],"r0":r[0],"f0":f[0],
                "p1":p[1],"r1":r[1],"f1":f[1]}

################################################################################
# 3) Simultaneous Multi‐View + Multi‐Task Model
################################################################################
class SimultaneousMultiViewMultiTaskNN(nn.Module):
    """
    Multi‐view + multi‐task with shared head ψ:
      L = BCE(mx, y) + BCE(mz, y)
        + λ_aux*(MSE(aux1)+MSE(aux2)+BCE(aux3))
        + λ_direct*MSE(s, s')
    """
    def __init__(self, dim_x, dim_z, hidden=64, lr=1e-3,
                 epochs=100, lambda_aux=0.3, lambda_direct=0.3):
        super().__init__()
        self.epochs         = epochs
        self.lambda_aux     = lambda_aux
        self.lambda_direct  = lambda_direct

        self.encoder_x = nn.Sequential(
            nn.Linear(dim_x, hidden), nn.ReLU()
        )
        self.encoder_z = nn.Sequential(
            nn.Linear(dim_z, hidden), nn.ReLU()
        )
        self.shared    = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4)
        )

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.opt = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, z):
        s, sp = self.encoder_x(x), self.encoder_z(z)
        out_x = self.shared(s)
        out_z = self.shared(sp)
        mx   = torch.sigmoid(out_x[:,0])
        a1   = out_x[:,1]
        a2   = out_x[:,2]
        a3   = torch.sigmoid(out_x[:,3])
        mz   = torch.sigmoid(out_z[:,0])
        return s, sp, mx, mz, a1, a2, a3

    def compute_loss(self, x, z, y, a1, a2, a3):
        s, sp, mx, mz, x1, x2, x3 = self.forward(x, z)
        l_main   = self.bce(mx, y) + self.bce(mz, y)
        l_aux    = self.mse(x1, a1) + self.mse(x2, a2) + self.bce(x3, a3)
        l_direct = self.mse(s, sp)
        return l_main + self.lambda_aux*l_aux + self.lambda_direct*l_direct

    def train_model(self, X, Z, y, a1, a2, a3,
                x_val=None, z_val=None, y_val=None,
                a1_val=None, a2_val=None, a3_val=None,
                early_stopping_patience=10, record_loss=False):
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.train()
            self.opt.zero_grad()
            loss = self.compute_loss(X, Z, y, a1, a2, a3)
            loss.backward()
            self.opt.step()

            if record_loss:
                train_loss_history.append(loss.item())

            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = self.compute_loss(x_val, z_val, y_val, a1_val, a2_val, a3_val)
                    if record_loss:
                        val_loss_history.append(val_loss.item())

                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            break

        if record_loss:
            return train_loss_history, val_loss_history

    def evaluate(self, X, y, Z):
        self.eval()
        with torch.no_grad():
            _, _, mx, mz, _, _, _ = self.forward(X, Z)
            pred = (mx + mz) / 2
        preds = (pred>=0.5).float().cpu().numpy()
        y_np  = y.cpu().numpy()
        # overall accuracy
        acc = (preds == y_np).mean()
        # per‐class accuracy
        acc0 = (preds[y_np==0] == 0).mean() if (y_np==0).any() else 0.0
        acc1 = (preds[y_np==1] == 1).mean() if (y_np==1).any() else 0.0
        try:
            auc = roc_auc_score(y_np, pred.cpu().numpy())
        except:
            auc = 0.0
        p, r, f, _ = precision_recall_fscore_support(y_np, preds, zero_division=0)
        return {
            "accuracy": acc,
            "accuracy_class0": acc0,
            "accuracy_class1": acc1,
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1]
        }

################################################################################
# 4) Simple Multi‐View MLP (view alignment loss)
################################################################################
class MultiViewMLP(nn.Module):
    """
    φ(x) branch vs β(z) branch, enforce latent alignment:
      L = BCE(ψ(s), y) + λ * MSE(s, s')
    """
    def __init__(self, dim_x, dim_z, hidden=32, num_layers_x=3, num_layers_z=2,
                 lr=1e-3, lambda_view=0.3, epochs=100):
        super().__init__()
        self.epochs      = epochs
        self.lambda_view = lambda_view

        # φ(x)
        layers = []
        d = dim_x
        for _ in range(num_layers_x):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.phi = nn.Sequential(*layers)

        # β(z)
        layers = []
        d = dim_z
        for _ in range(num_layers_z):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.beta = nn.Sequential(*layers)

        self.classifier = nn.Sequential(nn.Linear(hidden,1), nn.Sigmoid())
        self.opt        = optim.Adam(self.parameters(), lr=lr)
        self.bce        = nn.BCELoss()
        self.mse        = nn.MSELoss()
        print(f"[INIT] dim_z={dim_z}, dim_x={dim_x}")

    def forward(self, x, z):
        s  = self.phi(x)
        sp = self.beta(z)
        return self.classifier(s).view(-1), s, sp

    def train_model(self, X, Z, y,
                x_val=None, y_val=None, z_val=None,
                early_stopping_patience=10, record_loss=False):
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.train()
            self.opt.zero_grad()
            yh, s, sp = self.forward(X, Z)
            loss_main = self.bce(yh, y)
            loss = loss_main + self.lambda_view * self.mse(s, sp)
            loss.backward()
            self.opt.step()

            if record_loss:
                train_loss_history.append(loss_main.item())

            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    yh_val, s_val, sp_val = self.forward(x_val, z_val if z_val is not None else torch.zeros_like(x_val))
                    val_loss_main = self.bce(yh_val, y_val)
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

    def evaluate(self, X, y, Z=None):
        self.eval()
        with torch.no_grad():
            yh, s, sp = self.forward(X, Z if Z is not None else X*0)
        preds = (yh>=0.5).float().cpu().numpy()
        y_np  = y.cpu().numpy()
        p, r, f, _ = precision_recall_fscore_support(y_np, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_np, yh.cpu().numpy())
        except:
            auc = 0.0
        return {"auc":auc,
                "p0":p[0],"r0":r[0],"f0":f[0],
                "p1":p[1],"r1":r[1],"f1":f[1]}
