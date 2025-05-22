# # # # scripts/pairwise.py
# # # import torch
# # # from data.pairwise_data import PairwisePreprocessor
# # # from model.pairwise_model.soft import PairwiseDiabetesModel_Soft
# # # from model.pairwise_model.continuous import PairwiseDiabetesModel_Continuous
# # # from sklearn.model_selection import KFold
# # # from utils import set_seed, grid_search_cv_pairwise, run_experiments_pairwise
# # # import numpy as np
# # # import csv

# # # def main():
# # #     set_seed(42)
# # #     prep=PairwisePreprocessor('prompting/augmented_data_pairwise_8B.csv',
# # #                               'prompting/pairs.json', dataset_id=891)
# # #     grid_set, train_set, test_set=prep.preprocess()

# # #     grid={'margin':[1.0],'lambda_w':[0.3,0.5,0.7],'gamma':[1.0],
# # #           'hidden_dim':[32,64],'lr':[1e-3],'epochs':[100]}

# # #     for name,model in [('Soft',PairwiseDiabetesModel_Soft),
# # #                        ('Cont',PairwiseDiabetesModel_Continuous)]:
# # #         best,auc=grid_search_cv_pairwise(model,grid,grid_set)
# # #         print(f"{name} best: {best}, AUC={auc:.3f}")
# # #         results=run_experiments_pairwise(model,best,train_set,test_set)
# # #         agg={k:f"{np.mean([r[k] for r in results]):.2f}±{np.std([r[k] for r in results]):.2f}" for k in results[0]}
# # #         with open(f"results_pairwise_{name}.csv",'w',newline='') as f:
# # #             w=csv.writer(f); w.writerow(['Metric','Value']); w.writerows(agg.items())

# # # if __name__=='__main__': 
# # #     main()

# # # scripts/pairwise.py
# # import torch
# # from data.pairwise_data import PairwisePreprocessor
# # from model.pairwise_model.soft import PairwiseDiabetesModel_Soft
# # from model.pairwise_model.continuous import PairwiseDiabetesModel_Continuous
# # from sklearn.model_selection import KFold
# # from utils import set_seed, grid_search_cv_pairwise, run_experiments_pairwise
# # import numpy as np
# # import csv

# # def main():
# #     set_seed(42)
# #     feat_cols=[ 'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
# #         'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
# #         'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
# #         'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']
# #     prep=PairwisePreprocessor('prompting/augmented_data_pairwise_8B.csv',
# #                               'prompting/pairs.json',891,feat_cols,'Diabetes_binary')
# #     grid_set, train_set, test_set=prep.preprocess()

# #     grid={'margin':[1.0],'lambda_w':[0.3,0.5,0.7],'gamma':[1.0],
# #           'hidden_dim':[32,64],'lr':[1e-3],'epochs':[100]}

# #     for name,model in [('Soft',PairwiseDiabetesModel_Soft),
# #                        ('Cont',PairwiseDiabetesModel_Continuous)]:
# #         best,auc=grid_search_cv_pairwise(model,grid,grid_set)
# #         print(f"{name} best: {best}, AUC={auc:.3f}")
# #         results=run_experiments_pairwise(model,best,train_set,test_set)
# #         agg={k:f"{np.mean([r[k] for r in results]):.2f}±{np.std([r[k] for r in results]):.2f}" for k in results[0]}
# #         with open(f"results_pairwise_{name}.csv",'w',newline='') as f:
# #             w=csv.writer(f); w.writerow(['Metric','Value']); w.writerows(agg.items())

# # if __name__=='__main__': main()

# import random
# import csv
# import numpy as np
# import torch
# from sklearn.model_selection import KFold
# from data.pairwise_data import PairwisePreprocessor
# from model.pairwise_model.pretrain_finetune import PairwiseDiabetesModel_PretrainFineTune
# from model.pairwise_model.simultaneous import PairwiseDiabetesPredictor
# from utils import set_seed


# def grid_search_pretrain_finetune(model_cls, param_grid, grid_set, cv=3):
#     """
#     Grid search for pretrain+finetune model hyperparameters using KFold CV on grid_set.
#     grid_set: ((Xi, Xj, S, yi, yj)) pairs for pretraining and fine-tuning splits.
#     Returns best_params dict and best AUC.
#     """
#     Xi, Xj, S, yi, _ = grid_set
#     best_auc = -np.inf
#     best_params = None
#     keys = list(param_grid.keys())

#     for combo in np.array(np.meshgrid(*[param_grid[k] for k in keys])).T.reshape(-1, len(keys)):
#         params = dict(zip(keys, combo))
#         aucs = []
#         kf = KFold(n_splits=cv, shuffle=True, random_state=42)

#         for tr_idx, va_idx in kf.split(Xi):
#             # Split for pretrain and fine-tune
#             X1_tr, X1_va = Xi[tr_idx], Xi[va_idx]
#             X2_tr, X2_va = Xj[tr_idx], Xj[va_idx]
#             S_tr, S_va = S[tr_idx], S[va_idx]
#             y_tr, y_va = yi[tr_idx], yi[va_idx]

#             # Instantiate model with stage hyperparameters
#             model = model_cls(
#                 input_dim=Xi.shape[1],
#                 hidden_dim=int(params['hidden_dim']),
#                 lr=float(params['lr']),
#                 pre_epochs=int(params['pre_epochs']),
#                 fine_epochs=int(params['fine_epochs'])
#             )
#             # Pretrain encoder φ
#             model.pretrain(
#                 X1_tr, X2_tr, S_tr,
#                 margin=float(params['margin']),
#                 lambda_w=float(params['lambda_w']),
#                 gamma=float(params['gamma'])
#             )
#             # Fine-tune φ+ψ on supervised labels
#             model.finetune(X1_tr, y_tr)

#             # Evaluate on validation fold
#             metrics = model.evaluate(X1_va, y_va)
#             aucs.append(metrics['auc'])

#         avg_auc = float(np.mean(aucs))
#         if avg_auc > best_auc:
#             best_auc = avg_auc
#             best_params = params

#     return best_params, best_auc


# def run_experiments_pretrain_finetune(
#     model_cls, params, train_set, test_set,
#     runs=10, sample_size_per_class=250
# ):
#     """
#     Execute multiple runs for the pretrain+finetune model.
#     train_set: (Xi, Xj, S, yi, yj)
#     test_set: (X_test, y_test)
#     Returns list of metrics dicts per run.
#     """
#     Xi, Xj, S, yi, _ = train_set
#     X_test, y_test = test_set
#     metrics_list = []

#     # indices for balanced sampling
#     idx1 = (yi[:,0] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
#     idx0 = (yi[:,0] == 0).nonzero(as_tuple=True)[0].cpu().numpy()
#     test_pool_size = len(X_test)

#     for run in range(runs):
#         # Balanced sampling from train set
#         sel1 = np.random.choice(idx1, min(len(idx1), sample_size_per_class), replace=False)
#         sel0 = np.random.choice(idx0, min(len(idx0), sample_size_per_class), replace=False)
#         sel = np.concatenate([sel1, sel0])
#         np.random.shuffle(sel)

#         # Prepare run data
#         X1_run, X2_run = Xi[sel], Xj[sel]
#         S_run = S[sel]
#         y_run = yi[sel]

#         # Instantiate model
#         model = model_cls(
#             input_dim=Xi.shape[1],
#             hidden_dim=int(params['hidden_dim']),
#             lr=float(params['lr']),
#             pre_epochs=int(params['pre_epochs']),
#             fine_epochs=int(params['fine_epochs'])
#         )
#         # Pretraining
#         model.pretrain(
#             X1_run, X2_run, S_run,
#             margin=float(params['margin']),
#             lambda_w=float(params['lambda_w']),
#             gamma=float(params['gamma'])
#         )
#         # Fine-tuning
#         model.finetune(X1_run, y_run)

#         # Randomly sample test subset
#         test_idx = random.sample(range(test_pool_size), min(125, test_pool_size))
#         X_test_run = X_test[test_idx]
#         y_test_run = y_test[test_idx]

#         # Evaluate
#         metrics = model.evaluate(X_test_run, y_test_run)
#         print(f"Run {run}: {metrics}")
#         metrics_list.append(metrics)

#     return metrics_list


# def main():
#     set_seed(42)

#     feat_cols = [
#         'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
#         'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
#         'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
#         'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'
#     ]
#     prep = PairwisePreprocessor(
#         pairwise_csv_path='prompting/augmented_data_pairwise_70B.csv',
#         pairs_json_path='prompting/pairs.json',
#         dataset_id=891,
#         feature_cols=feat_cols,
#         target_col='Diabetes_binary'
#     )
#     # grid_set used for hyperparameter search, train_set/test_set for experiments
#     grid_set, train_set, test_set = prep.preprocess()

#     # Hyperparameter grid
#     param_grid = {
        # 'hidden_dim': [32, 64, 128, 256],
        # 'lr': [0.001, 0.01],
        # 'pre_epochs': [50, 100],
        # 'fine_epochs': [50, 100],
        # 'margin': [1.0],
        # 'lambda_w': [0.3, 0.5, 0.7],
        # 'gamma': [1.0, 0.0]
#     }

#     # Grid search
#     best_params, best_auc = grid_search_pretrain_finetune(
#         PairwiseDiabetesModel_PretrainFineTune,
#         # PairwiseDiabetesPredictor,
#         param_grid,
#         grid_set,
#         cv=3
#     )
#     print(f"Best params: {best_params}, Best AUC: {best_auc:.4f}")

#     # Run experiments
#     results = run_experiments_pretrain_finetune(
#         PairwiseDiabetesModel_PretrainFineTune,
#         # PairwiseDiabetesPredictor,
#         best_params,
#         train_set,
#         test_set,
#         runs=10,
#         sample_size_per_class=250
#     )

#     # Aggregate metrics
#     agg = {k: f"{np.mean([r[k] for r in results]):.2f}±{np.std([r[k] for r in results]):.2f}" 
#            for k in results[0]}
#     print("=== Final Aggregated Results ===")
#     for metric, value in agg.items():
#         print(f"{metric}: {value}")

#     # Save to CSV
#     with open('results_pretrain_finetune.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Metric','Value'])
#         for metric, value in agg.items():
#             writer.writerow([metric, value])


# if __name__ == '__main__':
#     main()




import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import json
import pandas as pd
import torch
import numpy as np
from ucimlrepo import fetch_ucirepo
import random
# utils.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import torch
import itertools
import random

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

def load_csv(path):
    return pd.read_csv(path)

class PairwisePreprocessor:
    """
    Builds pairwise dataset for diabetes: features, targets, and similarity.
    Returns:
      - grid_set: (Xi, Xj, S, yi, yj) for hyperparameter CV
      - train_set: (Xi, Xj, S, yi, yj) for final training
      - test_set: (X, y) single-instance test set
    """
    def __init__(self, pairwise_csv_path, pairs_json_path, dataset_id, feature_cols, target_col):
        self.pairwise_csv_path = pairwise_csv_path
        self.pairs_json_path = pairs_json_path
        self.dataset_id = dataset_id
        self.feature_cols = feature_cols
        self.target_col = target_col
        # fetch original
        data = fetch_ucirepo(id=dataset_id).data
        self.X_orig = data.features.copy()
        self.y_orig = data.targets

    def preprocess(self, grid_size_per_class=375):
        # load augmented pairwise
        df_aug = pd.read_csv(self.pairwise_csv_path)
        pairs = json.load(open(self.pairs_json_path))
        # build lookup
        orig = self.X_orig.copy()
        orig[self.target_col] = self.y_orig
        if 'index' not in orig.columns:
            orig = orig.reset_index().rename(columns={'index':'index'})
        lookup = orig.set_index('index')
        # build all pairs
        records = []
        for i_id, group in df_aug.groupby('index'):
            js = pairs.get(str(i_id), [])
            if len(js) < len(group):
                continue
            r_i = group.iloc[0]
            feat_i = r_i[self.feature_cols].values.astype(np.float32)
            tgt_i = r_i[self.target_col]
            for k, (_, r) in enumerate(group.iterrows()):
                try:
                    row_j = lookup.loc[js[k]]
                except KeyError:
                    continue
                feat_j = row_j[self.feature_cols].values.astype(np.float32)
                tgt_j = row_j[self.target_col]
                sim = r['similarity']
                # record pair
                rec = {f'{c}_i': feat_i[idx] for idx, c in enumerate(self.feature_cols)}
                rec.update({'target_i': tgt_i})
                rec.update({f'{c}_j': feat_j[idx] for idx, c in enumerate(self.feature_cols)})
                rec.update({'target_j': tgt_j, 'similarity': sim})
                records.append(rec)
        df = pd.DataFrame(records)
        before = len(df)
        df = df.dropna()
        after = len(df)
        print(f"Removed {before - after} rows due to NaN values.")
        # balanced grid for CV
        pos = df[df['target_i'] == 1]
        neg = df[df['target_i'] == 0]
        grid = pd.concat([
            pos.sample(min(len(pos), grid_size_per_class), random_state=42),
            neg.sample(min(len(neg), grid_size_per_class), random_state=42)
        ])
        train_df = df.drop(grid.index)
        # single-instance test pool
        test_orig = self.X_orig.copy()
        test_orig[self.target_col] = self.y_orig
        test_df = test_orig.drop(df['target_i'].index, errors='ignore')
        # convert to tensors
        def to_pair(df_pair):
            Xi = torch.tensor(df_pair[[f'{c}_i' for c in self.feature_cols]].values, dtype=torch.float)
            Xj = torch.tensor(df_pair[[f'{c}_j' for c in self.feature_cols]].values, dtype=torch.float)
            S  = torch.tensor(df_pair['similarity'].values, dtype=torch.float).view(-1,1)
            yi = torch.tensor(df_pair['target_i'].values, dtype=torch.float).view(-1,1)
            yj = torch.tensor(df_pair['target_j'].values, dtype=torch.float).view(-1,1)
            return Xi, Xj, S, yi, yj
        def to_single(df_single):
            X = torch.tensor(df_single[self.feature_cols].values, dtype=torch.float)
            y = torch.tensor(df_single[self.target_col].values, dtype=torch.float).view(-1,1)
            return X, y
        return to_pair(grid), to_pair(train_df), to_single(test_df)


def plot_loss_curve(train_losses, val_losses, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Total Loss")
    plt.plot(val_losses,   label="Validation Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def grid_search_cv_pairwise(model_cls, param_grid, grid_set, cv=3):
    """
    Grid search for pairwise model using classification AUC on grid_set.

    grid_set: tuple (Xi, Xj, S, yi, yj)
    param_grid: dict of hyperparameters including keys for init (hidden_dim, lr, epochs)
                and train_model (margin, lambda_w, gamma)
    Returns best_params and best_auc.
    """
    Xi, Xj, S, yi, yj = grid_set
    best_auc = -np.inf
    best_params = None
    keys = list(param_grid.keys())

    # init keys for constructor
    init_keys = ['hidden_dim', 'lr', 'epochs']
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        # separate init vs train args
        init_kwargs = {k: params[k] for k in init_keys if k in params}
        aucs = []
        for tr_idx, va_idx in kf.split(Xi):
            init_params = {k: params[k] for k in ['hidden_dim', 'lr', 'epochs']}
            train_params = {k: params[k] for k in ['margin', 'lambda_w', 'gamma']}

            model = model_cls(input_dim=Xi.shape[1], **init_params)
            model.train_model(
                Xi[tr_idx], Xj[tr_idx], S[tr_idx], yi[tr_idx], yj[tr_idx],
                margin=params.get('margin', 1.0),
                lambda_w=params.get('lambda_w', 0.5),
                gamma=params.get('gamma', 1.0),
                record_loss=False
            )
            # evaluate classification on Xi_val
            yhat = model.predict_proba(Xi[va_idx])
            try:
                aucs.append(roc_auc_score(yi[va_idx].cpu().numpy(), yhat.cpu().numpy()))
            except Exception:
                aucs.append(0.0)
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params
    return best_params, best_auc


def run_experiments_pairwise(model_cls, params, train_set, test_set, runs=10, sample_size=250):
    """
    Run multiple experiments:
      - balanced sampling on train_set pairs
      - train/val split on pairs
      - train_model with record_loss
      - test on random subset of test_set single instances
    Returns list of metrics dicts.
    """
    Xi, Xj, S, yi, yj = train_set
    X_test, y_test = test_set
    metrics_list = []
    # class indices for yi
    pos_idx = (yi[:,0]==1).nonzero(as_tuple=True)[0].cpu().numpy()
    neg_idx = (yi[:,0]==0).nonzero(as_tuple=True)[0].cpu().numpy()
    test_N = len(X_test)
    for run in range(runs):
        # balanced sampling of pairs
        sel1 = np.random.choice(pos_idx, min(len(pos_idx), sample_size), replace=False)
        sel0 = np.random.choice(neg_idx, min(len(neg_idx), sample_size), replace=False)
        sel = np.concatenate([sel1, sel0]); np.random.shuffle(sel)
        tr, va = sel[:int(0.8*len(sel))], sel[int(0.8*len(sel)):]
        
        # train
        init_params = {k: params[k] for k in ['hidden_dim', 'lr', 'epochs'] if k in params}
        train_params = {
            'margin': params.get('margin', 1.0),
            'lambda_w': params.get('lambda_w', 0.5),
            'gamma': params.get('gamma', 1.0),
        }

        model = model_cls(input_dim=Xi.shape[1], **init_params)
        train_losses, val_losses = model.train_model(
            Xi[tr], Xj[tr], S[tr], yi[tr], yj[tr],
            Xi[va], Xj[va], S[va], yi[va], yj[va],
            record_loss=True,
            **train_params
        )

        # plot losses
        plot_loss_curve(train_losses, val_losses, f"img/{model_cls.__name__}_run{run}.png")
        # test on single instances
        test_idx = random.sample(range(test_N), min(test_N,125))
        Xs, ys = X_test[test_idx], y_test[test_idx]
        metrics = model.evaluate(Xs, ys)
        metrics_list.append(metrics)
        print(f"Run {run}:", metrics)
    return metrics_list

from utils import set_seed
if __name__ == '__main__':
    # Korean: 시드 고정
    set_seed(42)
    feat_cols = [ 'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
        'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
        'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
        'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']
    # 데이터 전처리
    prep = PairwisePreprocessor(
        'prompting/augmented_data_pairwise_8B.csv',
        'prompting/pairs.json',
        891, feat_cols, 'Diabetes_binary'
    )
    grid_set, train_set, test_set = prep.preprocess()

    # 하이퍼파라미터 그리드
    grid = {
        'hidden_dim': [32, 64, 128, 256],
        'lr': [0.001, 0.01],
        'epochs': [50, 100],
        'margin': [1.0],
        'lambda_w': [0.3, 0.5, 0.7],
        'gamma': [1.0, 0.0]
    }
    # 그리드 서치
    best_params, best_auc = grid_search_cv_pairwise(
        PairwiseDiabetesPredictor, grid, grid_set
    )
    print(f"Best params: {best_params}, AUC={best_auc:.3f}")

    # 최종 실험 (10 runs)
    results = run_experiments_pairwise(
        PairwiseDiabetesPredictor, best_params, train_set, test_set, runs=10
    )
    # 결과 집계
    agg = {k: f"{np.mean([r[k] for r in results]):.2f} ± {np.std([r[k] for r in results]):.2f}" 
           for k in results[0]}
    # CSV 저장
    with open('results_pairwise.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric','Value'])
        writer.writerows(agg.items())
