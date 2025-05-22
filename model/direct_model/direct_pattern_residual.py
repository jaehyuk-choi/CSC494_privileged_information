# direct_model/direct_pattern_residual.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class DirectPatternResidual(nn.Module):
    """
    Direct Pattern Model with Residual Connections.

    - Encoder compresses X to intermediate h.
    - to_S compresses h to scalar S.
    - expand_S expands S back to vector.
    - Residual: combine expanded S + last encoder output (sum or concat).
    - Decoder predicts ŷ from combined representation.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        num_layers=3,
        num_decoder_layers=1,
        residual_type="concat",
        lr=0.001,
        epochs=300
    ):
        super().__init__()
        self.epochs = epochs
        self.residual_type = residual_type

        # build encoder layers
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.feature_layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ))

        # compression and expansion
        self.to_S = nn.Linear(hidden_dim, 1)   # compress to scalar S
        self.expand_S = nn.Linear(1, hidden_dim)  # expand back to vector

        # build decoder
        decoder_input = hidden_dim * 2 if residual_type == "concat" else hidden_dim
        layers = []
        for _ in range(num_decoder_layers - 1):
            layers += [nn.Linear(decoder_input, hidden_dim), nn.ReLU()]
            decoder_input = hidden_dim
        layers += [nn.Linear(decoder_input, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)

        # optimizer & losses
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_z = nn.MSELoss()
        self.loss_y = nn.BCELoss()

    def forward(self, x):
        # encode and collect residuals
        residuals = []
        h = x
        for layer in self.feature_layers:
            h = layer(h)
            residuals.append(h)

        # compress to S and expand
        S = self.to_S(h)            # [batch,1]
        expanded = self.expand_S(S) # [batch,hidden_dim]

        # combine residual + expanded S
        last = residuals[-1]
        if self.residual_type == "sum":
            combined = expanded + last
        else:
            combined = torch.cat([expanded, last], dim=1)

        # decode
        y_pred = self.decoder(combined)
        return S, y_pred

    def train_model(self, x, y, z, x_val=None, y_val=None, z_val=None, early_stopping=10, record_loss=False):
        """
        - Train with joint loss: L = MSE(S,z) + BCE(ŷ,y)
        - Supports optional validation & early stopping.
        """
        train_hist, val_hist = [], []
        best_val = float('inf')
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

            # validation
            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    _, yv = self.forward(x_val)
                    vl = self.loss_y(yv, y_val).item()
                    if record_loss: val_hist.append(vl)
                    if vl < best_val:
                        best_val, patience = vl, 0
                    else:
                        patience += 1
                        if patience >= early_stopping:
                            break

        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, x, y):
        """
        Evaluate on test set: returns dict with accuracy, auc, BCE loss,
        and per-class precision/recall/F1.
        """
        self.eval()
        with torch.no_grad():
            _, y_pred = self.forward(x)
            y_pred=y_pred.view(-1)
            y = y.view(-1)
            y_hat = (y_pred >= 0.5).float()
            acc = (y_hat == y).float().mean().item()
            try:
                auc = roc_auc_score(y.cpu(), y_pred.cpu())
            except:
                auc = 0.0
            losses = self.loss_y(y_pred, y).item()
            p, r, f, _ = precision_recall_fscore_support(y.cpu(), y_hat.cpu(), zero_division=0)
        return {
            "accuracy": acc,
            "auc": auc,
            "bce_loss": losses,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1]
        }
