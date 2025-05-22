# direct_model/direct_pattern_no_decomp.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class DirectPatternNoDecomp(nn.Module):
    """
    Direct Pattern Model WITHOUT explicit expansion layer.
    
    - Encoder extracts features.
    - to_S compresses to scalar S.
    - Residual: combine S + weighted sum of encoder outputs.
    - Decoder predicts ŷ.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        num_layers=3,
        num_decoder_layers=1,
        residual_type="concat",
        lr=0.001,
        epochs=100
    ):
        super().__init__()
        self.epochs = epochs
        self.residual_type = residual_type

        # encoder
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.feature_layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU()
            ))

        # learnable weights for residual
        self.alpha = nn.Parameter(torch.ones(num_layers))

        # compression
        self.to_S = nn.Linear(hidden_dim, 1)

        # decoder
        decoder_input = hidden_dim + 1 if residual_type == "concat" else hidden_dim
        layers = []
        for _ in range(num_decoder_layers - 1):
            layers += [nn.Linear(decoder_input, hidden_dim), nn.ReLU()]
            decoder_input = hidden_dim
        layers += [nn.Linear(decoder_input, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)

        # losses & optimizer
        self.loss_z = nn.MSELoss()
        self.loss_y = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        outputs = []
        h = x
        for layer in self.feature_layers:
            h = layer(h)
            outputs.append(h)
        weights = torch.softmax(self.alpha, dim=0)
        residual = sum(w * o for w, o in zip(weights, outputs))

        S = self.to_S(outputs[-1])  # [batch,1]

        if self.residual_type == "sum":
            combined = S + residual
        else:
            combined = torch.cat([S, residual], dim=1)

        y_pred = self.decoder(combined)
        return S, y_pred

    def train_model(self, x, y, z, x_val=None, y_val=None, z_val=None, early_stopping=10, record_loss=False):
        """
        Joint training: L = MSE(S,z) + BCE(ŷ,y).
        """
        train_hist, val_hist = [], []
        best_val, patience = float('inf'), 0

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            S_pred, y_pred = self.forward(x)
            loss = self.loss_z(S_pred, z) + self.loss_y(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if record_loss: train_hist.append(self.loss_y(y_pred, y).item())

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
                        if patience >= early_stopping: break
        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, x, y):
        """
        Test evaluation: accuracy, auc, BCE, per-class p/r/f.
        """
        self.eval()
        with torch.no_grad():
            _, y_pred = self.forward(x)
            y_pred = y_pred.view(-1)
            y = y.view(-1)
            y_hat = (y_pred >= 0.5).float()
            print(y_pred, y_hat, "+=", y)
            acc = (y_hat == y).float().mean().item()
            try: auc = roc_auc_score(y.cpu(), y_pred.cpu())
            except: auc = 0.0
            bce = self.loss_y(y_pred, y).item()
            p, r, f, _ = precision_recall_fscore_support(y.cpu(), y_hat.cpu(), zero_division=0)
        return {
            "accuracy": acc, "auc": auc, "bce_loss": bce,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1]
        }

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# class DirectPatternNoDecomp(nn.Module):
#     """
#     Direct Pattern Model WITHOUT explicit expansion layer.
    
#     - Encoder extracts features.
#     - to_S compresses to scalar S.
#     - Residual: combine S + weighted sum of encoder outputs.
#     - Decoder predicts ŷ.
#     """
#     def __init__(
#         self,
#         input_dim,
#         hidden_dim=32,
#         num_layers=3,
#         num_decoder_layers=1,
#         residual_type="concat",
#         lr=0.001,
#         epochs=100
#     ):
#         super().__init__()
#         self.epochs = epochs
#         self.residual_type = residual_type

#         # encoder
#         self.feature_layers = nn.ModuleList()
#         for i in range(num_layers):
#             in_dim = input_dim if i == 0 else hidden_dim
#             self.feature_layers.append(nn.Sequential(
#                 nn.Linear(in_dim, hidden_dim), nn.ReLU()
#             ))

#         # learnable weights for residual
#         self.alpha = nn.Parameter(torch.ones(num_layers))

#         # compression
#         self.to_S = nn.Linear(hidden_dim, 1)

#         # decoder
#         decoder_input = hidden_dim + 1 if residual_type == "concat" else hidden_dim
#         layers = []
#         for _ in range(num_decoder_layers - 1):
#             layers += [nn.Linear(decoder_input, hidden_dim), nn.ReLU()]
#             decoder_input = hidden_dim
#         layers += [nn.Linear(decoder_input, 1), nn.Sigmoid()]
#         self.decoder = nn.Sequential(*layers)

#         # losses & optimizer
#         self.loss_z = nn.MSELoss()
#         self.loss_y = nn.BCELoss()
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)

#     def forward(self, x):
#         outputs = []
#         h = x
#         for layer in self.feature_layers:
#             h = layer(h)
#             outputs.append(h)
#         weights = torch.softmax(self.alpha, dim=0)
#         residual = sum(w * o for w, o in zip(weights, outputs))

#         S = self.to_S(outputs[-1])  # [batch,1]

#         if self.residual_type == "sum":
#             combined = S + residual
#         else:
#             combined = torch.cat([S, residual], dim=1)

#         y_pred = self.decoder(combined)
#         return S, y_pred

#     def train_model(self, x, y, z, x_val=None, y_val=None, z_val=None, early_stopping=10, record_loss=False):
#         """
#         Joint training: L = MSE(S,z) + BCE(ŷ,y).
#         """
#         train_hist, val_hist = [], []
#         best_val, patience = float('inf'), 0

#         for epoch in range(self.epochs):
#             self.train()
#             self.optimizer.zero_grad()
#             S_pred, y_pred = self.forward(x)
#             loss = self.loss_z(S_pred, z) + self.loss_y(y_pred, y)
#             loss.backward()
#             self.optimizer.step()
#             if record_loss: train_hist.append(self.loss_y(y_pred, y).item())

#             if x_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     _, yv = self.forward(x_val)
#                     vl = self.loss_y(yv, y_val).item()
#                     if record_loss: val_hist.append(vl)
#                     if vl < best_val:
#                         best_val, patience = vl, 0
#                     else:
#                         patience += 1
#                         if patience >= early_stopping: break
#         return (train_hist, val_hist) if record_loss else None

#     def evaluate(self, x, y):
#         """
#         Test evaluation: accuracy, auc, BCE, per-class p/r/f.
#         """
#         self.eval()
#         with torch.no_grad():
#             _, y_pred = self.forward(x)
#             y_pred = y_pred.view(-1)
#             y_hat = (y_pred >= 0.5).float()
#             acc = (y_hat == y).float().mean().item()
#             try: auc = roc_auc_score(y.cpu(), y_pred.cpu())
#             except: auc = 0.0
#             bce = self.loss_y(y_pred, y).item()
#             p, r, f, _ = precision_recall_fscore_support(y.cpu(), y_hat.cpu(), zero_division=0)
#         return {
#             "accuracy": acc, "auc": auc, "bce_loss": bce,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1]
#         }