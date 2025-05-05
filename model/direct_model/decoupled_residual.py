import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class DirectDecoupledResidualModel(nn.Module):
    """
    Direct Residual Model with DECOUPLED training phases,
    now with early stopping in finetune phase.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=3,
                 num_decoder_layers=1, residual_type="concat",
                 lr=0.001, epochs=100):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.residual_type = residual_type

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim = input_dim
        for _ in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.ReLU()
            ))
            dim = hidden_dim

        # Residual weights
        self.alpha = nn.Parameter(torch.ones(num_layers))

        # Compression & expansion
        self.to_S = nn.Linear(hidden_dim, 1)
        self.expand_S = nn.Linear(1, hidden_dim)

        # Decoder
        dec_in = hidden_dim * 2 if residual_type == "concat" else hidden_dim
        layers = []
        for _ in range(num_decoder_layers - 1):
            layers += [nn.Linear(dec_in, hidden_dim), nn.ReLU()]
            dec_in = hidden_dim
        layers += [nn.Linear(dec_in, 1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*layers)

        # Loss functions
        self.loss_z = nn.MSELoss()
        self.loss_y = nn.BCELoss()

    def forward(self, x):
        outs = []
        h = x
        for lay in self.encoder_layers:
            h = lay(h)
            outs.append(h)
        w = torch.softmax(self.alpha, dim=0)
        resid = sum(wi * oi for wi, oi in zip(w, outs))

        S = self.to_S(outs[-1])
        exp_S = self.expand_S(S)

        if self.residual_type == "sum":
            combined = exp_S + resid
        else:
            combined = torch.cat([exp_S, resid], dim=1)

        y_hat = self.decoder(combined)
        return S, y_hat

    def pretrain_phase(self, x, z, epochs=None, lr=None):
        """
        Phase1: train encoder & to_S & expand_S & alpha to minimize MSE(S,z).
        """
        e = epochs or self.epochs
        lr0 = lr or self.lr
        opt = optim.Adam(
            list(self.encoder_layers.parameters()) +
            list(self.to_S.parameters()) +
            list(self.expand_S.parameters()) + [self.alpha],
            lr=lr0
        )
        self.train()
        for _ in range(e):
            opt.zero_grad()
            S, _ = self.forward(x)
            loss = self.loss_z(S, z)
            loss.backward()
            opt.step()

    def train_model(self, x, y, z,
                    x_val=None, y_val=None, z_val=None,
                    early_stopping=10, record_loss=False):
        """
        Full training: pretrain + finetune with early stopping.
        Returns train/val loss lists if record_loss=True.
        """
        # Phase1: pretrain
        self.pretrain_phase(x, z)

        # Freeze encoder components
        for p in self.encoder_layers.parameters(): p.requires_grad = False
        for p in self.to_S.parameters(): p.requires_grad = False
        for p in self.expand_S.parameters(): p.requires_grad = False
        self.alpha.requires_grad = False

        # Phase2: finetune decoder
        opt = optim.Adam(self.decoder.parameters(), lr=self.lr)
        best_val = float('inf')
        patience = 0
        train_hist, val_hist = [], []

        for epoch in range(self.epochs):
            # Training
            self.train()
            opt.zero_grad()
            _, y_pred = self.forward(x)
            loss_train = self.loss_y(y_pred, y)
            loss_train.backward()
            opt.step()

            if record_loss:
                train_hist.append(loss_train.item())

            # Validation & early stopping
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    _, y_val_pred = self.forward(x_val)
                    loss_val = self.loss_y(y_val_pred, y_val)
                if record_loss:
                    val_hist.append(loss_val.item())
                if loss_val.item() < best_val:
                    best_val = loss_val.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping:
                        break

        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, x, y):
        """
        Evaluate decoder on (x -> y_hat) after both phases.
        """
        self.eval()
        with torch.no_grad():
            _, y_hat = self.forward(x)
            y_hat=y_hat.view(-1)
            preds = (y_hat >= 0.5).float()
            try:
                auc = roc_auc_score(y.cpu(), y_hat.cpu())
            except:
                auc = 0.0
            bce = self.loss_y(y_hat, y).item()
            p, r, f, _ = precision_recall_fscore_support(
                y.cpu(), preds.cpu(), zero_division=0
            )
        return {
            "auc": auc,
            "bce_loss": bce,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1]
        }
