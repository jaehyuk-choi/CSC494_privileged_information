import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# -----------------------------------------------------------------------------
# 1. MultiTaskLogisticRegression
# -----------------------------------------------------------------------------
class MultiTaskLogisticRegression(nn.Module):
    """
    Multi-task Logistic Regression with main BCE loss and auxiliary MSE/BCE losses.
    """
    def __init__(self, input_dim, optimizer_type="adam", lr=0.01,
                 lambda_aux1=0.3, lambda_aux2=0.3, lambda_aux3=0.3,
                 epochs=300):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        self.main_head = nn.Sequential(nn.Linear(input_dim,1), nn.Sigmoid())
        self.aux1 = nn.Linear(input_dim,1)  # regression
        self.aux2 = nn.Linear(input_dim,1)  # regression
        self.aux3 = nn.Sequential(nn.Linear(input_dim,1), nn.Sigmoid())  # binary

        self.lr = lr
        self.epochs = epochs
        self.lambda_aux1 = lambda_aux1
        self.lambda_aux2 = lambda_aux2
        self.lambda_aux3 = lambda_aux3

        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

        self.main_loss_fn = nn.BCELoss()
        self.aux_mse = nn.MSELoss()
        self.aux_bce = nn.BCELoss()

    def forward(self, x):
        feat = self.shared(x)
        main = self.main_head(feat).view(-1)
        a1 = self.aux1(feat).view(-1)
        a2 = self.aux2(feat).view(-1)
        a3 = self.aux3(feat).view(-1)
        return main, a1, a2, a3

    def train_model(self, x, main_y, aux1_y=None, aux2_y=None, aux3_y=None,
                    x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
                    early_stopping_patience=10, record_loss=False):
        train_losses, val_losses = [], []
        best_val, patience = float('inf'), 0

        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            main_out, a1_out, a2_out, a3_out = self.forward(x)
            loss_main = self.main_loss_fn(main_out, main_y.view(-1))

            # auxiliary losses
            if aux1_y is not None:
                loss_a1 = self.aux_mse(a1_out, aux1_y.view(-1))
                loss_a2 = self.aux_mse(a2_out, aux2_y.view(-1))
                loss_a3 = self.aux_bce(a3_out, aux3_y.view(-1))
                loss = loss_main + self.lambda_aux1*loss_a1 + self.lambda_aux2*loss_a2 + self.lambda_aux3*loss_a3
            else:
                loss = loss_main

            loss.backward()
            self.optimizer.step()

            if record_loss:
                train_losses.append(loss_main.item())

            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    val_main, _, _, _ = self.forward(x_val)
                    v_loss = self.main_loss_fn(val_main, main_y_val.view(-1))
                if record_loss:
                    val_losses.append(v_loss.item())
                if v_loss.item() < best_val:
                    best_val, patience = v_loss.item(), 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break

        if record_loss:
            return train_losses, val_losses

    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            main_out, _, _, _ = self.forward(x)
            preds = (main_out >= 0.5).float().cpu().numpy()
            y_np = y.cpu().numpy()
            p, r, f, _ = precision_recall_fscore_support(y_np, preds, labels=[0,1], zero_division=0)
            auc = roc_auc_score(y_np, main_out.cpu().numpy())
            loss = self.main_loss_fn(main_out, y.view(-1)).item()
        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
            "bce_loss": loss
        }


# -----------------------------------------------------------------------------
# 2. MultiTaskNN (vanilla MLP)
# -----------------------------------------------------------------------------
class MultiTaskNN(nn.Module):
    """
    Multi-task Neural Network with configurable number of intermediate layers.
    
    Changes:
      - The train_model() method records only the main task BCE loss (ignoring auxiliary losses)
        each epoch and applies early stopping (patience=10) if no improvement.
      - The evaluate() method also computes the main task BCE Loss on the test set.
      - The learned α parameters (alpha_aux1, alpha_aux2, alpha_aux3) will be recorded later.
    """
    def __init__(self, input_dim, hidden_dim=16, num_layers=1,
                 optimizer_type="adam", lr=0.01, lambda_aux=0.3, epochs=300):
        super(MultiTaskNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lambda_aux = lambda_aux  # Scaling factor for auxiliary tasks

        # Feature extractor: intermediate layers (fully-connected + ReLU)
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU()
                ))
            else:
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ))
        
        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Main task head (binary classification)
        self.main_task = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary task heads
        self.aux_task1 = nn.Linear(hidden_dim, 1)
        self.aux_task2 = nn.Linear(hidden_dim, 1)
        self.aux_task3 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Learnable weights (α) for each auxiliary task
        self.alpha_aux1 = nn.Parameter(torch.ones(num_layers))
        self.alpha_aux2 = nn.Parameter(torch.ones(num_layers))
        self.alpha_aux3 = nn.Parameter(torch.ones(num_layers))
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Loss functions
        self.main_loss_fn = nn.BCELoss()      # For main task (binary classification)
        self.aux_loss_fn = nn.MSELoss()       # For auxiliary tasks 1 and 2 (regression)
        self.aux_task3_loss_fn = nn.BCELoss() # For auxiliary task 3 (binary classification)
    
    def forward(self, x):
        # Save outputs of each intermediate layer (for skip connections)
        intermediate_outputs = []
        h = x
        for layer in self.feature_layers:
            h = layer(h)
            intermediate_outputs.append(h)
        
        # Main task: last intermediate layer -> shared layer -> main head
        h_main = self.shared(intermediate_outputs[-1])
        main_out = self.main_task(h_main).view(-1)
        
        # Auxiliary tasks: weighted sum of intermediate outputs (using softmax on α)
        def weighted_sum(alpha, outputs):
            weights = torch.softmax(alpha, dim=0)
            combined = sum(w * out for w, out in zip(weights, outputs))
            return combined
        
        h_aux1 = self.shared(weighted_sum(self.alpha_aux1, intermediate_outputs))
        h_aux2 = self.shared(weighted_sum(self.alpha_aux2, intermediate_outputs))
        h_aux3 = self.shared(weighted_sum(self.alpha_aux3, intermediate_outputs))
        
        aux1_out = self.aux_task1(h_aux1).view(-1)
        aux2_out = self.aux_task2(h_aux2).view(-1)
        aux3_out = self.aux_task3(h_aux3).view(-1)
        
        return main_out, aux1_out, aux2_out, aux3_out

    def train_model(self, x, main_y, aux1_y=None, aux2_y=None, aux3_y=None,
                    x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
                    early_stopping_patience=30, record_loss=False):
        """
        Train the model using training data and optionally validation data.
        If validation data is provided, compute the main task BCE loss each epoch.
        Training stops early if the validation main loss does not improve for the given patience.
        If record_loss=True, returns train and validation main loss history.
        """
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            main_out, a1_out, a2_out, a3_out = self.forward(x)
            loss_main = self.main_loss_fn(main_out, main_y.view(-1))
            # Backward pass uses the total loss (including auxiliary losses) but we only record the main loss.
            if aux1_y is not None:
                loss_aux1 = self.aux_loss_fn(a1_out, aux1_y.view(-1))
                loss_aux2 = self.aux_loss_fn(a2_out, aux2_y.view(-1))
                loss_aux3 = self.aux_task3_loss_fn(a3_out, aux3_y.view(-1))
                total_loss = loss_main + self.lambda_aux * (loss_aux1 + loss_aux2 + loss_aux3)
            else:
                total_loss = loss_main
            
            total_loss.backward()
            self.optimizer.step()
            if record_loss:
                # Record only the main task BCE loss
                train_loss_history.append(loss_main.item())
            
            if x_val is not None and main_y_val is not None:
                self.eval()
                with torch.no_grad():
                    main_out_val, a1_out_val, a2_out_val, a3_out_val = self.forward(x_val)
                    val_loss_main = self.main_loss_fn(main_out_val, main_y_val.view(-1))
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
    
    def evaluate(self, x, y):
        """
        Evaluate the model on the test set.
        Returns a dictionary with AUC, precision, recall, F1 scores, and test BCE Loss.
        """
        self.eval()
        with torch.no_grad():
            main_out, _, _, _ = self.forward(x)
            preds = (main_out >= 0.5).float()
            metrics = precision_recall_fscore_support(
                y.cpu().numpy(), preds.cpu().numpy(), labels=[0,1], zero_division=0
            )
            auc = roc_auc_score(y.cpu().numpy(), main_out.cpu().numpy())
            p0, r0, f0 = metrics[0][0], metrics[1][0], metrics[2][0]
            p1, r1, f1 = metrics[0][1], metrics[1][1], metrics[2][1]
            bce_loss = self.main_loss_fn(main_out, y.view(-1))
        return {"auc": auc, "p0": p0, "r0": r0, "f0": f0, "p1": p1, "r1": r1, "f1": f1, "bce_loss": bce_loss.item()}

# class MultiTaskNN(nn.Module):
#     """
#     Multi-task MLP with configurable hidden_dim and num_layers.
#     Records only main BCE loss for plotting.
#     """
#     def __init__(self, input_dim, hidden_dim=16, num_layers=1,
#                  lr=0.01, lambda_aux=0.3, epochs=300):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.lambda_aux = lambda_aux
#         self.epochs = epochs

#         # build feature extractor
#         layers = []
#         in_dim = input_dim
#         self.feature_layers = nn.ModuleList()
#         for i in range(num_layers):
#             if i == 0:
#                 self.feature_layers.append(nn.Sequential(
#                     nn.Linear(input_dim, hidden_dim),
#                     nn.ReLU()
#                 ))
#             else:
#                 self.feature_layers.append(nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim),
#                     nn.ReLU()
#                 ))
            
#         # # build feature extractor
#         # layers = []
#         # in_dim = input_dim
#         # for _ in range(num_layers):
#         #     layers.append(nn.Linear(in_dim, hidden_dim))
#         #     layers.append(nn.ReLU())
#         #     in_dim = hidden_dim
#         # self.feature = nn.Sequential(*layers)

#         self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

#         # heads
#         self.main_head = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())
#         self.aux1 = nn.Linear(hidden_dim,1)
#         self.aux2 = nn.Linear(hidden_dim,1)
#         self.aux3 = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())

#         # learnable attention on layers (α)
#         self.alpha1 = nn.Parameter(torch.ones(num_layers))
#         self.alpha2 = nn.Parameter(torch.ones(num_layers))
#         self.alpha3 = nn.Parameter(torch.ones(num_layers))

#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         self.main_loss_fn = nn.BCELoss()
#         self.aux_mse = nn.MSELoss()
#         self.aux_bce = nn.BCELoss()

#     def forward(self, x):
#         # collect intermediate outputs
#         outs = []
#         h = x
#         for layer in self.feature_layers:
#             h = layer(h)
#             outs.append(h)

#         # main path
#         h_main = self.shared(outs[-1])
#         main_out = self.main_head(h_main).view(-1)

#         # weighted aux path
#         def weighted(alpha, seq):
#             w = torch.softmax(alpha, dim=0)
#             return sum(w[i]*seq[i] for i in range(len(seq)))

#         h1 = self.shared(weighted(self.alpha1, outs))
#         h2 = self.shared(weighted(self.alpha2, outs))
#         h3 = self.shared(weighted(self.alpha3, outs))

#         a1 = self.aux1(h1).view(-1)
#         a2 = self.aux2(h2).view(-1)
#         a3 = self.aux3(h3).view(-1)
#         return main_out, a1, a2, a3

#     def train_model(self, x, y_main, aux1_y=None, aux2_y=None, aux3_y=None,
#                     x_val=None, main_y_val=None,
#                     early_stopping_patience=10, record_loss=False):
#         train_losses, val_losses = [], []
#         best_val, patience = float('inf'), 0

#         for epoch in range(self.epochs):
#             self.train()
#             self.optimizer.zero_grad()
#             out_main, out1, out2, out3 = self.forward(x)
#             loss_main = self.main_loss_fn(out_main, y_main.view(-1))
#             break
#             if aux1_y is not None:
#                 l1 = self.aux_mse(out1, aux1_y.view(-1))
#                 l2 = self.aux_mse(out2, aux2_y.view(-1))
#                 l3 = self.aux_bce(out3, aux3_y.view(-1))
#                 loss = loss_main + self.lambda_aux*(l1+l2+l3)
#             else:
#                 loss = loss_main

#             loss.backward()
#             self.optimizer.step()

#             if record_loss:
#                 train_losses.append(loss_main.item())
#             # validation
#             if x_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     v_main, _, _, _ = self.forward(x_val)
#                     v_loss = self.main_loss_fn(v_main, main_y_val.view(-1))
#                 if record_loss:
#                     val_losses.append(v_loss.item())
#                 if v_loss.item() < best_val:
#                     best_val, patience = v_loss.item(), 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping_patience:
#                         break

#         if record_loss:
#             return train_losses, val_losses

#     def evaluate(self, x, y):
#         self.eval()
#         with torch.no_grad():
#             out_main, _, _, _ = self.forward(x)
#             preds = (out_main >= 0.5).float().cpu().numpy()
#             y_np = y.cpu().numpy()
#             p, r, f, _ = precision_recall_fscore_support(y_np, preds, labels=[0,1], zero_division=0)
#             auc = roc_auc_score(y_np, out_main.cpu().numpy())
#             loss = self.main_loss_fn(out_main, y.view(-1)).item()
#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#             "bce_loss": loss
#         }


# -----------------------------------------------------------------------------
# 3. Pre-train & Fine-tune (extended with validation & early-stopping)
# -----------------------------------------------------------------------------
class MultiTaskNN_PretrainFinetuneExtended(nn.Module):
    """
    Two-phase multi-task network:
    - Phase1: pre-train on auxiliary losses.
    - Phase2: fine-tune on main task with optional validation & early stopping.
    Records pretrain, finetune train/val losses if requested.
    """
    def __init__(self, input_dim, hidden_dim=16, num_layers=1,
                 lr_pre=0.01, lr_fine=0.005, lambda_aux=0.3,
                 pre_epochs=100, fine_epochs=100):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lambda_aux = lambda_aux
        self.pre_epochs = pre_epochs
        self.fine_epochs = fine_epochs

        # build feature extractor as before...
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature = nn.Sequential(*layers)
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # heads
        self.main = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())
        self.aux1 = nn.Linear(hidden_dim,1)
        self.aux2 = nn.Linear(hidden_dim,1)
        self.aux3 = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())

        # attention params
        self.alpha1 = nn.Parameter(torch.ones(num_layers))
        self.alpha2 = nn.Parameter(torch.ones(num_layers))
        self.alpha3 = nn.Parameter(torch.ones(num_layers))

        # losses
        self.main_loss_fn = nn.BCELoss()
        self.aux_mse = nn.MSELoss()
        self.aux_bce = nn.BCELoss()

        self.lr_pre = lr_pre
        self.lr_fine = lr_fine

    def forward(self, x):
        outs = []
        h = x
        for layer in self.feature:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                outs.append(h)
        shared_main = self.shared(outs[-1])
        main_out = self.main(shared_main).view(-1)

        def weighted(alpha, seq):
            w = torch.softmax(alpha, dim=0)
            return sum(w[i]*seq[i] for i in range(len(seq)))

        s1 = self.shared(weighted(self.alpha1, outs))
        s2 = self.shared(weighted(self.alpha2, outs))
        s3 = self.shared(weighted(self.alpha3, outs))

        a1 = self.aux1(s1).view(-1)
        a2 = self.aux2(s2).view(-1)
        a3 = self.aux3(s3).view(-1)
        return main_out, a1, a2, a3

    def train_pre_fine(self,
                      x_pre, y_main_pre, aux1_pre, aux2_pre, aux3_pre,
                      x_fine, y_fine,
                      record_loss=False, early_stopping_patience=10,
                      x_fine_val=None, y_fine_val=None):
        pre_losses, fine_losses, val_losses = [], [], []

        # --- Phase 1: pre-training on auxiliary tasks ---
        opt_pre = optim.Adam(
            list(self.feature.parameters()) +
            list(self.aux1.parameters()) +
            list(self.aux2.parameters()) +
            list(self.aux3.parameters()) +
            [self.alpha1, self.alpha2, self.alpha3],
            lr=self.lr_pre
        )
        for _ in range(self.pre_epochs):
            self.train()
            opt_pre.zero_grad()
            _, o1, o2, o3 = self.forward(x_pre)
            l1 = self.aux_mse(o1, aux1_pre.view(-1))
            l2 = self.aux_mse(o2, aux2_pre.view(-1))
            l3 = self.aux_bce(o3, aux3_pre.view(-1))
            loss_side = self.lambda_aux*(l1 + l2 + l3)
            loss_side.backward()
            opt_pre.step()
            if record_loss:
                pre_losses.append(loss_side.item())

        # freeze auxiliary parameters
        self.alpha1.requires_grad = False
        self.alpha2.requires_grad = False
        self.alpha3.requires_grad = False
        for p in self.aux1.parameters(): p.requires_grad=False
        for p in self.aux2.parameters(): p.requires_grad=False
        for p in self.aux3.parameters(): p.requires_grad=False

        # --- Phase 2: fine-tuning on main task ---
        opt_fine = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr_fine)
        best_val, patience = float('inf'), 0

        for epoch in range(self.fine_epochs):
            self.train()
            opt_fine.zero_grad()
            out_main, _, _, _ = self.forward(x_fine)
            loss_main = self.main_loss_fn(out_main, y_fine.view(-1))
            loss_main.backward()
            opt_fine.step()
            if record_loss:
                fine_losses.append(loss_main.item())

            # early stopping on validation
            if x_fine_val is not None:
                self.eval()
                with torch.no_grad():
                    v_main, _, _, _ = self.forward(x_fine_val)
                    v_loss = self.main_loss_fn(v_main, y_fine_val.view(-1))
                if record_loss:
                    val_losses.append(v_loss.item())
                if v_loss.item() < best_val:
                    best_val, patience = v_loss.item(), 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break

        if record_loss:
            return pre_losses, fine_losses, val_losses

    def train_model(
        self,
        x, main_y,
        aux1=None, aux2=None, aux3=None,
        x_val=None, main_y_val=None,
        aux1_val=None, aux2_val=None, aux3_val=None,
        early_stopping_patience=10, record_loss=False
    ):
        # delegate to your two‑phase trainer
        return self.train_pre_fine(
            # pre‑training phase on side‑information
            x_pre=x,
            y_main_pre=main_y,
            aux1_pre=aux1,
            aux2_pre=aux2,
            aux3_pre=aux3,
            # fine‑tuning phase on main‑task
            x_fine=x,
            y_fine=main_y,
            # record losses?
            record_loss=record_loss,
            # correct early‐stop name
            early_stopping_patience=early_stopping_patience,
            # pass along validation set for fine‑tuning
            x_fine_val=x_val,
            y_fine_val=main_y_val
        )

    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            out_main, _, _, _ = self.forward(x)
            preds = (out_main >= 0.5).float().cpu().numpy()
            y_np = y.cpu().numpy()
            p, r, f, _ = precision_recall_fscore_support(y_np, preds, labels=[0,1], zero_division=0)
            auc = roc_auc_score(y_np, out_main.cpu().numpy())
            loss = self.main_loss_fn(out_main, y.view(-1)).item()
        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
            "bce_loss": loss
        }


# -----------------------------------------------------------------------------
# 4. Decoupled Procedure
# -----------------------------------------------------------------------------
class MultiTaskNN_Decoupled(nn.Module):
    """
    Decoupled multi-task network:
    - Phase1: optimize auxiliary losses φ & β.
    - Phase2: freeze φ & β, then optimize main head ψ.
    """
    def __init__(self, input_dim, hidden_dim=16, num_layers=1,
                 lr=0.01, lambda_aux=0.3,
                 pre_epochs=100, main_epochs=100):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.pre_epochs = pre_epochs
        self.main_epochs = main_epochs

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.feature = nn.Sequential(*layers)
        self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.main = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())
        self.aux1 = nn.Linear(hidden_dim,1)
        self.aux2 = nn.Linear(hidden_dim,1)
        self.aux3 = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())
        self.alpha1 = nn.Parameter(torch.ones(num_layers))
        self.alpha2 = nn.Parameter(torch.ones(num_layers))
        self.alpha3 = nn.Parameter(torch.ones(num_layers))

        self.main_loss_fn = nn.BCELoss()
        self.aux_mse = nn.MSELoss()
        self.aux_bce = nn.BCELoss()

        self.lr = lr

    def forward(self, x):
        outs = []
        h = x
        for layer in self.feature:
            h = layer(h)
            if isinstance(layer, nn.Linear):
                outs.append(h)
        shared_main = self.shared(outs[-1])
        main_out = self.main(shared_main).view(-1)

        def weighted(alpha, seq):
            w = torch.softmax(alpha, dim=0)
            return sum(w[i]*seq[i] for i in range(len(seq)))

        s1 = self.shared(weighted(self.alpha1, outs))
        s2 = self.shared(weighted(self.alpha2, outs))
        s3 = self.shared(weighted(self.alpha3, outs))
        a1 = self.aux1(s1).view(-1)
        a2 = self.aux2(s2).view(-1)
        a3 = self.aux3(s3).view(-1)
        return main_out, a1, a2, a3
    
    def train_model(
        self,
        x, y_main, aux1_y, aux2_y, aux3_y,
        x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
        early_stopping_patience=10,
        record_loss=False
    ):
        pre_losses, main_losses, val_losses = [], [], []

        # Phase1: auxiliary optimization
        opt_pre = optim.Adam(
            list(self.feature.parameters()) +
            list(self.aux1.parameters()) +
            list(self.aux2.parameters()) +
            list(self.aux3.parameters()) +
            [self.alpha1, self.alpha2, self.alpha3],
            lr=self.lr
        )
        for _ in range(self.pre_epochs):
            self.train()
            opt_pre.zero_grad()
            _, o1, o2, o3 = self.forward(x)
            loss_side = self.lambda_aux*( 
                self.aux_mse(o1, aux1_y.view(-1)) +
                self.aux_mse(o2, aux2_y.view(-1)) +
                self.aux_bce(o3, aux3_y.view(-1))
            )
            loss_side.backward()
            opt_pre.step()
            if record_loss:
                pre_losses.append(loss_side.item())

        # freeze φ & β
        for p in self.feature.parameters(): p.requires_grad=False
        for p in self.shared.parameters():  p.requires_grad=False
        for p in self.aux1.parameters():    p.requires_grad=False
        for p in self.aux2.parameters():    p.requires_grad=False
        for p in self.aux3.parameters():    p.requires_grad=False
        self.alpha1.requires_grad=self.alpha2.requires_grad=self.alpha3.requires_grad=False

        # Phase2: fine‑tune on main task with validation
        opt_main = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        best_val, patience = float('inf'), 0

        for _ in range(self.main_epochs):
            self.train()
            opt_main.zero_grad()
            out_main, _, _, _ = self.forward(x)
            loss_main = self.main_loss_fn(out_main, y_main.view(-1))
            loss_main.backward()
            opt_main.step()
            if record_loss:
                main_losses.append(loss_main.item())

            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    v_main, _, _, _ = self.forward(x_val)
                    v_loss = self.main_loss_fn(v_main, main_y_val.view(-1))
                if record_loss:
                    val_losses.append(v_loss.item())
                if v_loss.item() < best_val:
                    best_val, patience = v_loss.item(), 0
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        break
            self.train()

        if record_loss:
            print(main_losses, pre_losses)
            return main_losses, val_losses

    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            out_main, _, _, _ = self.forward(x)
            preds = (out_main >= 0.5).float().cpu().numpy()
            y_np = y.cpu().numpy()
            p, r, f, _ = precision_recall_fscore_support(y_np, preds, labels=[0,1], zero_division=0)
            auc = roc_auc_score(y_np, out_main.cpu().numpy())
            loss = self.main_loss_fn(out_main, y.view(-1)).item()
        return {
            "auc": auc,
            "p0": p[0], "r0": r[0], "f0": f[0],
            "p1": p[1], "r1": r[1], "f1": f[1],
            "bce_loss": loss
        }

# class MultiTaskNN_Decoupled(nn.Module):
#     """
#     Decoupled multi-task network:
#     - Phase1: optimize auxiliary losses φ & β.
#     - Phase2: freeze φ & β, then optimize main head ψ.
#     """
#     def __init__(self, input_dim, hidden_dim=16, num_layers=1,
#                  lr=0.01, lambda_aux=0.3,
#                  pre_epochs=100, main_epochs=100):
#         super().__init__()
#         self.lambda_aux = lambda_aux
#         self.pre_epochs = pre_epochs
#         self.main_epochs = main_epochs
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim

#         self.feature = nn.ModuleList()
#         for i in range(num_layers):
#             if i == 0:
#                 self.feature.append(nn.Sequential(
#                     nn.Linear(input_dim, hidden_dim),
#                     nn.ReLU()
#                 ))
#             else:
#                 self.feature.append(nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim),
#                     nn.ReLU()
#                 ))

#         self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
#         self.main = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())
#         self.aux1 = nn.Linear(hidden_dim,1)
#         self.aux2 = nn.Linear(hidden_dim,1)
#         self.aux3 = nn.Sequential(nn.Linear(hidden_dim,1), nn.Sigmoid())

#         self.alpha1 = nn.Parameter(torch.ones(num_layers))
#         self.alpha2 = nn.Parameter(torch.ones(num_layers))
#         self.alpha3 = nn.Parameter(torch.ones(num_layers))

#         self.main_loss_fn = nn.BCELoss()
#         self.aux_mse = nn.MSELoss()
#         self.aux_bce = nn.BCELoss()

#         self.lr = lr

#     def forward(self, x):
#         outs = []
#         h = x
#         for layer in self.feature:
#             h = layer(h)
#             outs.append(h)
#         shared_main = self.shared(outs[-1])
#         main_out = self.main(shared_main).view(-1)

#         def weighted(alpha, seq):
#             weights = torch.softmax(alpha, dim=0)
#             combined = sum(w * out for w, out in zip(weights, seq))
#             return combined

#         s1 = self.shared(weighted(self.alpha1, outs))
#         s2 = self.shared(weighted(self.alpha2, outs))
#         s3 = self.shared(weighted(self.alpha3, outs))
#         a1 = self.aux1(s1).view(-1)
#         a2 = self.aux2(s2).view(-1)
#         a3 = self.aux3(s3).view(-1)
#         return main_out, a1, a2, a3

#     def train_model(
#         self,
#         x, y_main, aux1_y, aux2_y, aux3_y,
#         x_val=None, main_y_val=None,
#         early_stopping_patience=10,
#         record_loss=False
#     ):
#         pre_losses, main_losses, val_losses = [], [], []

#         # Phase1: auxiliary optimization (기존 그대로)
#         opt_pre = optim.Adam(
#             list(self.feature.parameters()) +
#             list(self.aux1.parameters()) +
#             list(self.aux2.parameters()) +
#             list(self.aux3.parameters()) +
#             [self.alpha1, self.alpha2, self.alpha3],
#             lr=self.lr
#         )
#         for _ in range(self.pre_epochs):
#             self.train()
#             opt_pre.zero_grad()
#             _, o1, o2, o3 = self.forward(x)
#             loss_side = self.lambda_aux*( 
#                 self.aux_mse(o1, aux1_y.view(-1)) +
#                 self.aux_mse(o2, aux2_y.view(-1)) +
#                 self.aux_bce(o3, aux3_y.view(-1))
#             )
#             loss_side.backward()
#             opt_pre.step()
#             if record_loss:
#                 pre_losses.append(loss_side.item())

#         # freeze φ & β
#         for p in self.feature.parameters(): p.requires_grad=False
#         for p in self.shared.parameters():  p.requires_grad=False
#         for p in self.aux1.parameters():    p.requires_grad=False
#         for p in self.aux2.parameters():    p.requires_grad=False
#         for p in self.aux3.parameters():    p.requires_grad=False
#         self.alpha1.requires_grad=self.alpha2.requires_grad=self.alpha3.requires_grad=False

#         # Phase2: fine‑tune on main task with validation
#         opt_main = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
#         best_val, patience = float('inf'), 0

#         for _ in range(self.main_epochs):
#             self.train()
#             opt_main.zero_grad()
#             out_main, _, _, _ = self.forward(x)
#             loss_main = self.main_loss_fn(out_main, y_main.view(-1))
#             loss_main.backward()
#             opt_main.step()
#             if record_loss:
#                 main_losses.append(loss_main.item())

#             if x_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     v_main, _, _, _ = self.forward(x_val)
#                     v_loss = self.main_loss_fn(v_main, main_y_val.view(-1))
#                 if record_loss:
#                     val_losses.append(v_loss.item())
#                 if v_loss.item() < best_val:
#                     best_val, patience = v_loss.item(), 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping_patience:
#                         break
#             self.train()

#         if record_loss:
#             return pre_losses, main_losses, val_losses

#     def evaluate(self, x, y):
#         self.eval()
#         with torch.no_grad():
#             out_main, _, _, _ = self.forward(x)
#             preds = (out_main >= 0.5).float().cpu().numpy()
#             y_np = y.cpu().numpy()
#             p, r, f, _ = precision_recall_fscore_support(y_np, preds, labels=[0,1], zero_division=0)
#             auc = roc_auc_score(y_np, out_main.cpu().numpy())
#             loss = self.main_loss_fn(out_main, y.view(-1)).item()
#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#             "bce_loss": loss
#         }