import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class DirectDecoupledResidualNoDecomp(nn.Module):
    """
    Decoupled Residual Model WITHOUT explicit expansion,
    with early stopping in finetune phase.
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=3,
                 num_decoder_layers=1, residual_type="concat",
                 lr=0.001, epochs=100):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.residual_type = residual_type

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        dim = input_dim
        for _ in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.ReLU()
            ))
            dim = hidden_dim

        # Residual weights
        self.alpha = nn.Parameter(torch.ones(num_layers))

        # Compression to scalar S
        self.to_S = nn.Linear(hidden_dim, 1)

        # Decoder network
        dec_in = hidden_dim + 1 if residual_type == "concat" else hidden_dim
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

        if self.residual_type == "sum":
            comb = S + resid
        else:
            comb = torch.cat([S, resid], dim=1)

        y_hat = self.decoder(comb)
        return S, y_hat

    def pretrain_phase(self, x, z, epochs=None, lr=None):
        """
        Phase 1: train encoder, to_S, and alpha to minimize MSE(S, z).
        """
        e = epochs or self.epochs
        lr0 = lr or self.lr
        opt = optim.Adam(
            list(self.encoder_layers.parameters()) +
            list(self.to_S.parameters()) + [self.alpha],
            lr=lr0
        )
        self.train()
        for _ in range(e):
            opt.zero_grad()
            S, _ = self.forward(x)
            loss = self.loss_z(S, z)
            loss.backward()
            opt.step()

    def train_model(self, x, y, z, x_val=None, y_val=None, z_val=None,
                    early_stopping=10, record_loss=False):
        """
        Full training: pretrain + finetune with early stopping on BCE loss.

        Returns:
            (train_hist, val_hist) if record_loss else None
        """
        # Phase 1: pretrain
        self.pretrain_phase(x, z)

        # Freeze encoder components
        for p in self.encoder_layers.parameters():
            p.requires_grad = False
        for p in self.to_S.parameters():
            p.requires_grad = False
        self.alpha.requires_grad = False

        # Phase 2: finetune decoder with early stopping
        opt = optim.Adam(self.decoder.parameters(), lr=self.lr)
        train_hist, val_hist = [], []
        best_val = float('inf')
        patience = 0

        for epoch in range(self.epochs):
            # Training step
            self.train()
            opt.zero_grad()
            _, yh = self.forward(x)
            loss = self.loss_y(yh, y)
            loss.backward()
            opt.step()

            if record_loss:
                train_hist.append(loss.item())

            # Validation and early stopping
            if x_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    _, yv = self.forward(x_val)
                    val_loss = self.loss_y(yv, y_val)
                if record_loss:
                    val_hist.append(val_loss.item())
                if val_loss.item() < best_val:
                    best_val = val_loss.item()
                    patience = 0
                else:
                    patience += 1
                    if patience >= early_stopping:
                        break

        return (train_hist, val_hist) if record_loss else None

    def evaluate(self, x, y):
        """
        Evaluate on test set: returns dict with AUC, BCE loss,
        and per-class precision/recall/F1.
        """
        self.eval()
        with torch.no_grad():
            _, y_pred = self.forward(x)
            y = y.view(-1)
            y_pred=y_pred.view(-1)
            preds = (y_pred >= 0.5).float()
            try:
                auc = roc_auc_score(y.cpu(), y_pred.cpu())
            except:
                auc = 0.0
            bce = self.loss_y(y_pred, y).item()
            p, r, f, _ = precision_recall_fscore_support(
                y.cpu(), preds.cpu(), zero_division=0
            )
        return {
            "auc": auc,
            "bce_loss": bce,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1]
        } 

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
            y = y.view(-1)
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
