# model/pairwise_model/pretrain_finetune.py

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

class PairwiseDiabetesModel_PretrainFineTune(nn.Module):
    """
    Two-stage training:
      1) Pretrain encoder φ on pairwise similarity/dissimilarity loss
      2) Fine-tune both φ and ψ on supervised BCE loss
    """

    def __init__(self, input_dim, hidden_dim=32, lr=1e-3, pre_epochs=100, fine_epochs=50):
        super().__init__()
        self.pre_epochs = pre_epochs
        self.fine_epochs = fine_epochs

        # φ: shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # ψ: classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # optimizers will be created in each stage
        self.lr = lr
        self.bce = nn.BCELoss()

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.forward_once(x)
        y = self.classifier(z)
        return z, y

    def pretrain(self, X1, X2, S, margin=1.0, lambda_w=0.5, gamma=1.0):
        """
        Stage 1: train φ only, using pairwise regularization loss.
        ψ is not used in this stage.
        """
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        for _ in range(self.pre_epochs):
            opt.zero_grad()
            z1 = self.encoder(X1)
            z2 = self.encoder(X2)
            dist = (z1 - z2).norm(dim=1, keepdim=True)
            S_norm = S / 10.0

            L_sim = (S_norm * dist**2).mean()
            L_dissim = ((1 - S_norm) * torch.clamp(margin - dist, min=0)**2).mean()
            loss = gamma * (lambda_w * L_sim + (1 - lambda_w) * L_dissim)

            loss.backward()
            opt.step()

    def finetune(self, X, y):
        """
        Stage 2: train both φ and ψ on supervised BCE loss.
        """
        params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.fine_epochs):
            opt.zero_grad()
            z = self.encoder(X)
            yp = self.classifier(z)
            yp = torch.clamp(yp, 1e-7, 1 - 1e-7)
            y_cl = torch.clamp(y, 1e-7, 1 - 1e-7)
            loss = self.bce(yp, y_cl)
            loss.backward()
            opt.step()

    def evaluate(self, X, y):
        """
        Evaluate on single-instance X → φ → ψ →ŷ
        Returns auc, precision/recall/f1, class-wise accuracy.
        """
        self.eval()
        with torch.no_grad():
            z = self.encoder(X)
            yp = self.classifier(z)
            yp_cl = (yp >= 0.5).float()

            try:
                auc = roc_auc_score(y.cpu(), yp.cpu())
            except:
                auc = 0.0

            p, r, f, _ = precision_recall_fscore_support(
                y.cpu(), yp_cl.cpu(), labels=[0,1], zero_division=0
            )
            acc0 = ((yp_cl==0)&(y==0)).sum().item() / max((y==0).sum().item(),1)
            acc1 = ((yp_cl==1)&(y==1)).sum().item() / max((y==1).sum().item(),1)

        return {
            'auc': auc,
            'p0': p[0], 'r0': r[0], 'f0': f[0], 'acc0': acc0,
            'p1': p[1], 'r1': r[1], 'f1': f[1], 'acc1': acc1
        }
