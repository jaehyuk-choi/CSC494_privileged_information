# model/pairwise_model/soft.py
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class PairwiseDiabetesModel_Soft(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, lr=1e-3, epochs=100):
        super().__init__()
        self.epochs = epochs
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.bce = nn.BCELoss()

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
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
        train_hist, val_hist = [], []
        for _ in range(self.epochs):
            # --- training step ---
            self.train(); self.opt.zero_grad()
            z1, z2, yp = self.forward(X1, X2)
            Lc = self.bce(yp, y1)
            dist = (z1 - z2).norm(dim=1, keepdim=True)
            Sn = S / 10
            Ls = (Sn * dist**2).mean()
            Ld = ((1-Sn) * torch.clamp(margin-dist, 0)**2).mean()
            loss = Lc + gamma * (lambda_w * Ls + (1-lambda_w) * Ld)
            loss.backward(); self.opt.step()

            if record_loss:
                train_hist.append(loss.item())
                if X1_val is not None:
                    # --- validation step ---
                    self.eval()
                    with torch.no_grad():
                        z1v, z2v, ypv = self.forward(X1_val, X2_val)
                        Lcv = self.bce(ypv, y1_val)
                        distv = (z1v - z2v).norm(dim=1, keepdim=True)
                        Sv = S_val / 10
                        Lsv = (Sv * distv**2).mean()
                        Ldv = ((1-Sv) * torch.clamp(margin-distv,0)**2).mean()
                        val_hist.append(Lcv.item() + gamma * (lambda_w * Lsv + (1-lambda_w) * Ldv))
        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, X1, X2, S, y1, y2, margin=1.0, lambda_w=0.5, gamma=1.0):
        self.eval()
        with torch.no_grad():
            z1, z2, yp = self.forward(X1, X2)
            yp_cl = (yp>=0.5).float()
            auc = roc_auc_score(y1.cpu(), yp.cpu()) if len(y1)>1 else 0.0
            p, r, f, _ = precision_recall_fscore_support(y1.cpu(), yp_cl.cpu(), zero_division=0)
            acc0 = ((yp_cl==0)&(y1==0)).sum().item() / max((y1==0).sum().item(),1)
            acc1 = ((yp_cl==1)&(y1==1)).sum().item() / max((y1==1).sum().item(),1)
            return {
                'auc':auc, 'p0':p[0],'r0':r[0],'f0':f[0], 'acc0':acc0,
                'p1':p[1],'r1':r[1],'f1':f[1], 'acc1':acc1
            }
