# model/pairwise_model/continuous_single_predict.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class PairwiseDiabetesPredictor(nn.Module):
    """
    Continuous similarity-regularized classifier for diabetes.
    Trains φ and ψ jointly using pairwise similarity, 
    but at test time predicts from a single input x.
    """
    def __init__(self, input_dim, hidden_dim=32, lr=1e-3, epochs=100):
        super().__init__()
        self.epochs = epochs

        # φ: Shared encoder that maps x -> embedding z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # ψ: Classifier head that maps z -> probability ŷ
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # optimizer and classification loss
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.bce = nn.BCELoss()

    def forward_once(self, x):
        """Compute embedding z = φ(x)."""
        return self.encoder(x)

    def forward(self, x1, x2):
        """
        Forward pass for a pair (x1, x2):
        - z1, z2: embeddings for x1 and x2
        - ŷ1: predicted probability for x1
        """
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        y1_hat = self.classifier(z1)
        return z1, z2, y1_hat

    def train_model(
        self,
        X1, X2, S, y1, y2,
        X1_val=None, X2_val=None, S_val=None, y1_val=None, y2_val=None,
        margin=1.0, lambda_w=0.5, gamma=1.0, record_loss=False
    ):
        """
        Train φ and ψ with:
         - classification loss L_cls = BCE(ψ(φ(x1)), y1)
         - L_sim = mean(S_norm * ||z1-z2||^2)
         - L_dissim = mean((1-S_norm) * clamp(margin - ||z1-z2||,0)^2)
         - Total loss = L_cls + γ [λ L_sim + (1-λ) L_dissim]
        If record_loss=True, returns (train_losses, val_losses).
        """
        train_losses, val_losses = [], []

        for epoch in range(self.epochs):
            self.train()
            self.opt.zero_grad()

            # forward on training pairs
            z1, z2, y1_hat = self.forward(X1, X2)
            y1_hat = torch.clamp(y1_hat, 1e-7, 1-1e-7)
            y1_clamped = torch.clamp(y1, 1e-7, 1-1e-7)

            # classification loss
            loss_cls = self.bce(y1_hat, y1_clamped)

            # continuous pairwise regularization
            dist = (z1 - z2).norm(dim=1, keepdim=True)
            S_norm = S / 10.0
            loss_sim = (S_norm * dist**2).mean()
            loss_dissim = ((1 - S_norm) * F.relu(margin - dist)**2).mean()

            loss = loss_cls + gamma * (lambda_w * loss_sim + (1 - lambda_w) * loss_dissim)
            loss.backward()
            self.opt.step()

            if record_loss:
                train_losses.append(loss.item())
                # validation loss
                if X1_val is not None:
                    self.eval()
                    with torch.no_grad():
                        z1v, z2v, y1v_hat = self.forward(X1_val, X2_val)
                        y1v_hat = torch.clamp(y1v_hat, 1e-7, 1-1e-7)
                        y1v_clamped = torch.clamp(y1_val, 1e-7, 1-1e-7)
                        loss_cls_v = self.bce(y1v_hat, y1v_clamped)
                        dist_v = (z1v - z2v).norm(dim=1, keepdim=True)
                        S_norm_v = S_val / 10.0
                        loss_sim_v = (S_norm_v * dist_v**2).mean()
                        loss_dissim_v = ((1 - S_norm_v) * F.relu(margin - dist_v)**2).mean()
                        val_losses.append((loss_cls_v + gamma * (lambda_w * loss_sim_v + (1 - lambda_w) * loss_dissim_v)).item())

        if record_loss:
            return train_losses, val_losses

    def predict_proba(self, x):
        """
        Predict probability ŷ = ψ(φ(x)) for single instances at test time.
        """
        self.eval()
        with torch.no_grad():
            z = self.forward_once(x)
            return self.classifier(z)

    def evaluate(self, X, y):
        """
        Evaluate on single-instance test set (X, y):
        returns AUC, precision, recall, f1, and class-wise accuracy.
        """
        self.eval()
        with torch.no_grad():
            y_hat = self.predict_proba(X)
            y_pred = (y_hat >= 0.5).float()

            # AUC
            try:
                auc = roc_auc_score(y.cpu(), y_hat.cpu())
            except ValueError:
                auc = 0.0

            # precision, recall, f1
            p, r, f, _ = precision_recall_fscore_support(
                y.cpu(), y_pred.cpu(), labels=[0,1], zero_division=0
            )
            # class-wise accuracy
            acc0 = ((y_pred==0) & (y==0)).sum().item() / max((y==0).sum().item(),1)
            acc1 = ((y_pred==1) & (y==1)).sum().item() / max((y==1).sum().item(),1)

            return {
                'auc': auc,
                'p0': p[0], 'r0': r[0], 'f0': f[0], 'acc0': acc0,
                'p1': p[1], 'r1': r[1], 'f1': f[1], 'acc1': acc1
            }
