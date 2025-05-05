# model/direct_model/combined_multitask.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class CombinedMultiTaskNN(nn.Module):
    """
    Combined Direct + Multi‑Task Model.
    """
    def __init__(self, input_dim, hidden_dim=16, num_layers=1,
                 lr=0.01, lambda_aux=0.3, lambda_direct=0.3, epochs=300):
        super().__init__()
        self.epochs = epochs
        self.lambda_aux = lambda_aux
        self.lambda_direct = lambda_direct

        # Encoder
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i==0 else hidden_dim
            self.feature_layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ))

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        # Heads
        self.main_task = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())
        self.aux1 = nn.Linear(hidden_dim,1)
        self.aux2 = nn.Linear(hidden_dim,1)
        self.aux3 = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())

        # Direct scalar head
        self.to_scalar = nn.Linear(hidden_dim,1)

        # Weights for aux residuals
        self.alpha1 = nn.Parameter(torch.ones(num_layers))
        self.alpha2 = nn.Parameter(torch.ones(num_layers))
        self.alpha3 = nn.Parameter(torch.ones(num_layers))

        # Loss & optimizer
        self.main_loss = nn.BCELoss()
        self.aux_loss = nn.MSELoss()
        self.aux3_loss = nn.BCELoss()
        self.direct_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        inter = []
        h = x
        for lay in self.feature_layers:
            h = lay(h)
            inter.append(h)

        # scalar S
        S_vec = inter[-1]
        S = self.to_scalar(S_vec).view(-1)

        # main out
        main_h = self.shared(S_vec)
        y_main = self.main_task(main_h).view(-1)

        # weighted-sum helper
        def wsum(alpha):
            w = torch.softmax(alpha, dim=0)
            return sum(wi*oi for wi,oi in zip(w, inter))

        # aux outs
        h1 = self.shared(wsum(self.alpha1))
        h2 = self.shared(wsum(self.alpha2))
        h3 = self.shared(wsum(self.alpha3))
        y1 = self.aux1(h1).view(-1)
        y2 = self.aux2(h2).view(-1)
        y3 = self.aux3(h3).view(-1)

        return S, y_main, y1, y2, y3

    def train_model(self, x, y, aux1, aux2, aux3,
                    x_val=None, y_val=None, a1_val=None, a2_val=None, a3_val=None,
                    early_stopping=10, record_loss=False):
        tr_hist, val_hist = [], []
        best_val, patience = float('inf'), 0

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            S, ym, a1, a2, a3 = self.forward(x)
            loss_m = self.main_loss(ym, y.view(-1))
            loss_d = self.direct_loss(S, aux1.view(-1))
            loss_1 = self.aux_loss(a1, aux1.view(-1))
            loss_2 = self.aux_loss(a2, aux2.view(-1))
            loss_3 = self.aux3_loss(a3, aux3.view(-1))
            total = loss_m + self.lambda_direct*loss_d + self.lambda_aux*(loss_1+loss_2+loss_3)
            total.backward()
            self.optimizer.step()
            if record_loss:
                tr_hist.append(loss_m.item())

            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    _, ymv, _, _, _ = self.forward(x_val)
                    vl = self.main_loss(ymv, y_val.view(-1)).item()
                if record_loss: val_hist.append(vl)
                if vl < best_val:
                    best_val, patience = vl, 0
                else:
                    patience += 1
                    if patience >= early_stopping: break

        return (tr_hist, val_hist) if record_loss else None

    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            _, y_pred, _, _, _ = self.forward(x)
            y_pred=y_pred.view(-1)
            y_hat = (y_pred>=0.5).float()
            try: auc = roc_auc_score(y.cpu(), ym.cpu())
            except: auc=0.0
            bce = self.main_loss(y_pred, y.view(-1)).item()
            p, r, f, _ = precision_recall_fscore_support(y.cpu(), y_hat.cpu(), zero_division=0)
        return {
            "auc": auc, "bce_loss": bce,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1]
        }
