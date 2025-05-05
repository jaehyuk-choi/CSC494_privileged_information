# model/pairwise_model/continuous.py

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class PairwiseDiabetesModel_Continuous(nn.Module):
    """
    Continuous similarity regularization pairwise model.
    Records train and val loss per epoch.
    """
    def __init__(self, input_dim, hidden_dim=32, lr=1e-3, epochs=100):
        super().__init__()
        self.epochs = epochs

        # Shared encoder φ: maps input features to an embedding vector.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Classification head ψ: predicts diabetes probability from the first embedding.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Optimizer and loss
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.bce = nn.BCELoss()

    def forward_once(self, x):
        """Compute embedding for one input."""
        return self.encoder(x)

    def forward(self, x1, x2):
        """
        Forward pass for a pair:
        - z1, z2: embeddings
        - y1: predicted probability for x1
        """
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        y1 = self.classifier(z1)
        return z1, z2, y1

    def train_model(
        self,
        X1, X2, S, y1, y2,
        X1_val=None, X2_val=None, S_val=None, y1_val=None, y2_val=None,
        margin=1.0, lambda_w=0.5, gamma=1.0, record_loss=False
    ):
        """
        Train the model:
          - L_cls: BCE loss on y1 vs y1_true
          - L_sim: mean(S_norm * ||z1-z2||^2)
          - L_dissim: mean((1-S_norm) * clamp(margin - ||z1-z2||, 0)^2)
          - Total: L_cls + γ * (λ * L_sim + (1-λ) * L_dissim)

        If record_loss=True and validation data provided, also compute val loss each epoch.
        Returns (train_losses, val_losses) if record_loss else None.
        """
        train_hist, val_hist = [], []

        for _ in range(self.epochs):
            # --- training step ---
            self.train()
            self.opt.zero_grad()

            z1, z2, yp = self.forward(X1, X2)
            # clamp predictions for numerical stability
            yp = torch.clamp(yp, 1e-7, 1 - 1e-7)
            y1_cl = torch.clamp(y1, 1e-7, 1 - 1e-7)
            Lc = self.bce(yp, y1_cl)

            # pairwise regularization
            dist = (z1 - z2).norm(dim=1, keepdim=True)
            S_norm = S / 10.0
            Ls = (S_norm * dist**2).mean()
            Ld = ((1 - S_norm) * torch.clamp(margin - dist, min=0)**2).mean()

            loss = Lc + gamma * (lambda_w * Ls + (1 - lambda_w) * Ld)
            loss.backward()
            self.opt.step()

            if record_loss:
                train_hist.append(loss.item())
                # --- validation step ---
                if X1_val is not None:
                    self.eval()
                    with torch.no_grad():
                        z1v, z2v, ypv = self.forward(X1_val, X2_val)
                        ypv = torch.clamp(ypv, 1e-7, 1 - 1e-7)
                        y1v_cl = torch.clamp(y1_val, 1e-7, 1 - 1e-7)
                        Lcv = self.bce(ypv, y1v_cl)
                        distv = (z1v - z2v).norm(dim=1, keepdim=True)
                        Sv = S_val / 10.0
                        Lsv = (Sv * distv**2).mean()
                        Ldv = ((1 - Sv) * torch.clamp(margin - distv, min=0)**2).mean()
                        val_hist.append(Lcv.item() + gamma * (lambda_w * Lsv + (1 - lambda_w) * Ldv))

        if record_loss:
            return train_hist, val_hist
        else:
            return None

    def evaluate(self, X1, X2, S, y1, y2, margin=1.0, lambda_w=0.5, gamma=1.0):
        """
        Evaluate the model and return metrics:
        - auc
        - precision, recall, f1 for each class
        - accuracy per class (acc0, acc1)
        """
        self.eval()
        with torch.no_grad():
            z1, z2, yp = self.forward(X1, X2)
            yp_cl = (yp >= 0.5).float()

            # AUC
            try:
                auc = roc_auc_score(y1.cpu(), yp.cpu())
            except ValueError:
                auc = 0.0

            # Precision / recall / f1
            p, r, f, _ = precision_recall_fscore_support(
                y1.cpu(), yp_cl.cpu(), labels=[0, 1], zero_division=0
            )

            # Class-wise accuracy
            acc0 = ((yp_cl == 0) & (y1 == 0)).sum().item() / max((y1 == 0).sum().item(), 1)
            acc1 = ((yp_cl == 1) & (y1 == 1)).sum().item() / max((y1 == 1).sum().item(), 1)

            return {
                'auc': auc,
                'p0': p[0], 'r0': r[0], 'f0': f[0], 'acc0': acc0,
                'p1': p[1], 'r1': r[1], 'f1': f[1], 'acc1': acc1
            }
