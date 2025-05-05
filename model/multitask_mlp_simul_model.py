# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# class MultiTaskNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim=16, num_layers=1, lr=0.01, lambda_aux=0.3, epochs=300):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.epochs = epochs
#         self.lambda_aux = lambda_aux

#         self.feature_layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU())
#             for i in range(num_layers)
#         ])

#         self.shared = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

#         self.main_task = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
#         self.aux_task1 = nn.Linear(hidden_dim, 1)
#         self.aux_task2 = nn.Linear(hidden_dim, 1)
#         self.aux_task3 = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

#         self.alpha_aux1 = nn.Parameter(torch.ones(num_layers))
#         self.alpha_aux2 = nn.Parameter(torch.ones(num_layers))
#         self.alpha_aux3 = nn.Parameter(torch.ones(num_layers))

#         self.optimizer = optim.Adam(self.parameters(), lr=lr)

#         self.main_loss_fn = nn.BCELoss()
#         self.aux_loss_fn = nn.MSELoss()
#         self.aux_task3_loss_fn = nn.BCELoss()

#     def forward(self, x):
#         inter_outs = []
#         h = x
#         for layer in self.feature_layers:
#             h = layer(h)
#             inter_outs.append(h)

#         def weighted_sum(alpha, outs):
#             weights = torch.softmax(alpha, dim=0)
#             return sum(w * o for w, o in zip(weights, outs))

#         h_main = self.shared(inter_outs[-1])
#         main_out = self.main_task(h_main).view(-1)

#         aux1_out = self.aux_task1(self.shared(weighted_sum(self.alpha_aux1, inter_outs))).view(-1)
#         aux2_out = self.aux_task2(self.shared(weighted_sum(self.alpha_aux2, inter_outs))).view(-1)
#         aux3_out = self.aux_task3(self.shared(weighted_sum(self.alpha_aux3, inter_outs))).view(-1)

#         return main_out, aux1_out, aux2_out, aux3_out

#     def train_model(self, x, main_y, aux1_y=None, aux2_y=None, aux3_y=None,
#                     x_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
#                     early_stopping=10, record_loss=False):
#         train_loss_history, val_loss_history = [], []
#         best_val = float('inf'); patience = 0

#         for epoch in range(self.epochs):
#             self.train()
#             self.optimizer.zero_grad()
#             m, a1, a2, a3 = self.forward(x)
#             loss_main = self.main_loss_fn(m, main_y.view(-1))
#             if aux1_y is not None:
#                 loss_aux1 = self.aux_loss_fn(a1, aux1_y.view(-1))
#                 loss_aux2 = self.aux_loss_fn(a2, aux2_y.view(-1))
#                 loss_aux3 = self.aux_task3_loss_fn(a3, aux3_y.view(-1))
#                 total_loss = loss_main + self.lambda_aux * (loss_aux1 + loss_aux2 + loss_aux3)
#             else:
#                 total_loss = loss_main
#             total_loss.backward(); self.optimizer.step()
#             if record_loss: train_loss_history.append(loss_main.item())

#             if x_val is not None and main_y_val is not None:
#                 self.eval()
#                 with torch.no_grad():
#                     m_val, _, _, _ = self.forward(x_val)
#                     val_loss = self.main_loss_fn(m_val, main_y_val.view(-1))
#                     if record_loss: val_loss_history.append(val_loss.item())
#                 if val_loss.item() < best_val:
#                     best_val = val_loss.item(); patience = 0
#                 else:
#                     patience += 1
#                     if patience >= early_stopping: break

#         if record_loss:
#             return train_loss_history, val_loss_history

#     def evaluate(self, x, y):
#         self.eval()
#         with torch.no_grad():
#             main_out, _, _, _ = self.forward(x)
#             preds = (main_out >= 0.5).float()
#             bce_loss = self.main_loss_fn(main_out, y.view(-1)).item()
#             y_np, preds_np = y.cpu().numpy(), preds.cpu().numpy()
#             auc = roc_auc_score(y_np, main_out.cpu().numpy())
#             p, r, f, _ = precision_recall_fscore_support(y_np, preds_np, zero_division=0)
#         return {
#             "auc": auc,
#             "p0": p[0], "r0": r[0], "f0": f[0],
#             "p1": p[1], "r1": r[1], "f1": f[1],
#             "bce_loss": bce_loss
#         }
