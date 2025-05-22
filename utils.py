# utils.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import torch
import itertools
import inspect
import random
import pandas as pd

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_loss_curve(train_losses, val_losses, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train BCE Loss")
    plt.plot(val_losses, label="Validation BCE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation BCE Loss")
    plt.legend()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def grid_search_cv(model_class, param_grid, X, y, cv=3, is_multitask=False,
                   aux1=None, aux2=None, aux3=None):
    """
    Perform grid search using ROC AUC.
    For multi-task models, only main task validation is used.
    """
    best_auc = -1.0
    best_params = None
    keys = list(param_grid.keys())

    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        aucs = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            x_tr, x_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if is_multitask:
                aux1_tr, aux2_tr, aux3_tr = aux1[train_idx], aux2[train_idx], aux3[train_idx]

            sig = inspect.signature(model_class.__init__).parameters
            # pass input_dim if required
            if 'input_dim' in sig:
                model = model_class(input_dim=X.shape[1], **params)
            else:
                model = model_class(**params)

            if is_multitask:
                model.train_model(
                    x_tr, y_tr, aux1_tr, aux2_tr, aux3_tr,
                    x_val=x_val, main_y_val=y_val,
                    early_stopping_patience=10,
                    record_loss=False
                )
            else:
                model.train_model(
                    x_tr, y_tr,
                    x_val=x_val, main_y_val=y_val,
                    early_stopping_patience=10,
                    record_loss=False
                )

            with torch.no_grad():
                out = model.forward(x_val)[0]
                aucs.append(roc_auc_score(y_val.cpu().numpy(), out.cpu().numpy()))

        avg_auc = float(np.mean(aucs))
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_params = params

    return best_params, best_auc

def grid_search_cv_basline(model_class, param_grid, X, y, cv=3,
                   aux1=None, aux2=None, aux3=None):
    best_auc = -1.0
    best_params = None
    keys = list(param_grid.keys())

    for values in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        aucs = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):
            x_tr, x_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            sig = inspect.signature(model_class.__init__).parameters
            # detect whether __init__ expects input_dim
            if 'input_dim' in sig:
                model = model_class(input_dim=X.shape[1], **params)
            else:
                model = model_class(**params)
                
            model.train_model(
                x_tr, y_tr,
                x_val=x_val, y_val=y_val,
                early_stopping=10
            )
            
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    out = model.forward(x_val)
                else:
                    # classical model
                    proba = model.model.predict_proba(x_val.numpy())[:, 1]
                    out = torch.from_numpy(proba).float()
                auc = roc_auc_score(y_val.cpu().numpy(), out.cpu().numpy())
                aucs.append(auc)

        avg_auc = float(np.mean(aucs))
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_params = params

    return best_params, best_auc

def grid_search_direct(model_class, param_grid, X, Y, Z, cv=3, early_stopping=10):
    """
    Perform grid search for direct pattern models.

    Args:
        model_class: class, expects __init__(input_dim, **params) and train_model / evaluate methods.
        param_grid: dict, keys are hyperparameter names, values are lists of candidate values.
        X: torch.Tensor of shape (N, D), input features for grid search.
        Y: torch.Tensor of shape (N,) or (N,1), main labels.
        Z: torch.Tensor of shape (N,) or (N,1), side-information labels.
        cv: int, number of cross-validation folds.
        early_stopping: int, patience for early stopping in train_model.

    Returns:
        best_params: dict of hyperparameters producing highest mean AUC.
        best_auc: float, corresponding average AUC.
    """
    best_auc, best_params = -np.inf, None
    keys = list(param_grid.keys())

    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        aucs = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for tr_idx, va_idx in kf.split(X):
            x_tr, x_va = X[tr_idx], X[va_idx]
            y_tr, y_va = Y[tr_idx], Y[va_idx]
            z_tr, z_va = Z[tr_idx], Z[va_idx]

            # Instantiate and train the model
            model = model_class(input_dim=X.shape[1], **params)
            model.train_model(
                x_tr, y_tr, z_tr,
                x_val=x_va, y_val=y_va, z_val=z_va,
                early_stopping=early_stopping,
                record_loss=False
            )
            # Obtain predictions on validation set
            _ , y_pred = model.forward(x_va)
            try:
                aucs.append(roc_auc_score(y_va.cpu().numpy(), y_pred.detach().cpu().numpy()))
            except Exception:
                aucs.append(0.0)

        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc, best_params = mean_auc, params

    return best_params, best_auc

def grid_search_cv_pairwise(model_cls, param_grid, grid_set, cv=3):
    Xi, Xj, S, yi, yj = grid_set
    best_params, best_score = None, float('-inf')
    for combo in __import__('itertools').product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        scores=[]
        kf=KFold(n_splits=cv, shuffle=True, random_state=42)
        for tr,va in kf.split(Xi):
            init = {k:params[k] for k in ['hidden_dim','lr','epochs'] if k in params}
            m = model_cls(input_dim=Xi.shape[1], **init)
            m.train_model(Xi[tr],Xj[tr],S[tr],yi[tr],yj[tr],
                          Xi[va],Xj[va],S[va],yi[va],yj[va],
                          margin=params.get('margin',1.0),
                          lambda_w=params.get('lambda_w',0.5),
                          gamma=params.get('gamma',1.0),
                          record_loss=False)
            scores.append(m.evaluate(Xi[va],Xj[va],S[va],yi[va],yj[va])['auc'])
        avg=np.mean(scores)
        if avg>best_score: best_score, best_params=avg, params
    return best_params, best_score

# def run_experiments(model_class, method_name, num_runs, best_params,
#                     train_x_pool, train_y_pool, test_x_pool, test_y_pool,
#                     is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
#     metrics_list = []
#     for run in range(1, num_runs + 1):
#         idx1 = train_y_pool.nonzero(as_tuple=True)[0]
#         idx0 = (train_y_pool == 0).nonzero(as_tuple=True)[0]
#         sampled_idx = torch.cat([
#             idx1[torch.randperm(len(idx1))[:250]],
#             idx0[torch.randperm(len(idx0))[:250]]
#         ])
#         x_train, y_train = train_x_pool[sampled_idx], train_y_pool[sampled_idx]

#         if is_multitask:
#             aux1 = train_aux1_pool[sampled_idx]
#             aux2 = train_aux2_pool[sampled_idx]
#             aux3 = train_aux3_pool[sampled_idx]

#         split = int(0.8 * len(x_train))
#         perm = torch.randperm(len(x_train))
#         tr_idx, val_idx = perm[:split], perm[split:]

#         sig = inspect.signature(model_class.__init__).parameters
#         if 'input_dim' in sig:
#             model = model_class(input_dim=x_train.shape[1], **best_params)
#         else:
#             model = model_class(**best_params)

#         if is_multitask:
#             train_losses, val_losses = model.train_model(
#                 x_train[tr_idx], y_train[tr_idx],
#                 aux1[tr_idx], aux2[tr_idx], aux3[tr_idx],
#                 x_val=x_train[val_idx], main_y_val=y_train[val_idx],
#                 early_stopping_patience=10, record_loss=True
#             )
#         else:
#             train_losses, val_losses = model.train_model(
#                 x_train[tr_idx], y_train[tr_idx],
#                 x_val=x_train[val_idx], y_val=y_train[val_idx],
#                 early_stopping_patience=10, record_loss=True
#             )

#         plot_loss_curve(train_losses, val_losses, f"img/{method_name}_run{run}.png")

#         # Randomly select 125 samples from the test pool
#         test_pool_size = len(test_x_pool)
#         test_indices = random.sample(range(test_pool_size), 125)
#         x_test_run = test_x_pool[test_indices]
#         y_test_run = test_y_pool[test_indices]
#         metrics = model.evaluate(x_test_run, y_test_run)
#         print(f"Run {run}: {metrics}")
#         metrics_list.append(metrics)

#     return metrics_list

# def grid_search_multiview(model_class, param_grid, X, Z, y, cv=3):
#     """
#     Grid search over param_grid for multi-view models using ROC AUC on main task.
#     Returns best_params dict and corresponding average AUC.
#     """
#     best_auc = -np.inf
#     best_params = None
#     keys = list(param_grid.keys())
#     for combo in itertools.product(*(param_grid[k] for k in keys)):
#         params = dict(zip(keys, combo))
#         aucs = []
#         kf = KFold(n_splits=cv, shuffle=True, random_state=42)

#         for tr_idx, va_idx in kf.split(X):
#             x_tr, z_tr, y_tr = X[tr_idx], Z[tr_idx], y[tr_idx]
#             x_va, z_va, y_va = X[va_idx], Z[va_idx], y[va_idx]

#             # prepare init args
#             sig = inspect.signature(model_class.__init__).parameters
#             init_kwargs = {}
#             # add dimension args
#             if 'dim_x' in sig:
#                 init_kwargs['dim_x'] = X.shape[1]
#             elif 'input_size_x' in sig:
#                 init_kwargs['input_size_x'] = X.shape[1]
#             if 'dim_z' in sig:
#                 print(Z.shape, init_kwargs)
#                 init_kwargs['dim_z'] = Z.shape[1]
#             elif 'input_size_z' in sig:
#                 init_kwargs['input_size_z'] = Z.shape[1]
#             # fill hyperparameters
#             for k, v in params.items():
#                 if k in sig:
#                     init_kwargs[k] = v
#             model = model_class(**init_kwargs)

#             if hasattr(model, 'pretrain_aux'):
#                 model.pretrain_aux(z_tr, y_tr)
#                 print("[INFO] ran pretrain_aux on auxiliary branch")

#             # then main training
#             try:
#                 model.train_model(x_tr, y_tr, z_tr, record_loss=True)
#             except TypeError:
#                 model.train_model(x_tr, z_tr, y_tr, record_loss=True)

#             # predict on validation
#             model.eval()
#             with torch.no_grad():
#                 out = model.forward(x_va, z_va)[0]
#             aucs.append(roc_auc_score(y_va.cpu().numpy(), out.cpu().numpy()))

#         avg_auc = np.mean(aucs)
#         if avg_auc > best_auc:
#             best_auc = avg_auc
#             best_params = params

#     return best_params, best_auc

# def run_experiments_multiview(
#     model_class,
#     method_name,
#     num_runs,
#     best_params,
#     train_X,
#     train_Z,
#     train_y,
#     test_X,
#     test_Z,
#     test_y
# ):
#     """
#     Run experiments for a multi-view model:
#     - Balanced sampling of train
#     - Train/val split (80/20)
#     - Record train & val loss, plot per run
#     - Evaluate on held-out test
#     """
#     metrics_list = []

#     # prepare indices for balanced sampling
#     y_np = train_y.cpu().numpy()
#     idx_pos = np.where(y_np == 1)[0].tolist()
#     idx_neg = np.where(y_np == 0)[0].tolist()

#     for run in range(1, num_runs + 1):
#         # sample up to 250 per class
#         sampled1 = random.sample(idx_pos, min(250, len(idx_pos)))
#         sampled0 = random.sample(idx_neg, min(250, len(idx_neg)))
#         idx_train = sampled1 + sampled0

#         x_pool = train_X[idx_train]
#         z_pool = train_Z[idx_train]
#         y_pool = train_y[idx_train]

#         # train/val split
#         perm = torch.randperm(len(x_pool))
#         split = int(0.8 * len(perm))
#         tr_idx, va_idx = perm[:split], perm[split:]
#         x_tr, z_tr, y_tr = x_pool[tr_idx], z_pool[tr_idx], y_pool[tr_idx]
#         x_va, z_va, y_va = x_pool[va_idx], z_pool[va_idx], y_pool[va_idx]

#         # instantiate model
#         sig = inspect.signature(model_class.__init__).parameters
#         init_kwargs = {}
#         if 'dim_x' in sig:
#             init_kwargs['dim_x'] = X.shape[1]
#         elif 'input_size_x' in sig:
#             init_kwargs['input_size_x'] = X.shape[1]
#         if 'dim_z' in sig:
#             init_kwargs['dim_z'] = Z.shape[1]
#         elif 'input_size_z' in sig:
#             init_kwargs['input_size_z'] = Z.shape[1]
#         for k, v in best_params.items():
#             if k in sig:
#                 init_kwargs[k] = v
#         model = model_class(**init_kwargs)

#         # train with recording
#         res = None
#         if hasattr(model, 'pretrain_aux'):
#             model.pretrain_aux(z_tr, y_tr)
#             print("[INFO] ran pretrain_aux on auxiliary branch")

#         # then main training
#         try:
#             model.train_model(x_tr, y_tr, z_tr, record_loss=True)
#         except TypeError:
#             model.train_model(x_tr, z_tr, y_tr, record_loss=True)
#         # try:
#         #     res = model.train_model(
#         #         x_tr, y_tr, z_tr,
#         #         record_loss=True,
#         #         x_val=x_va, y_val=y_va, z_val=z_va
#         #     )
#         # except TypeError:
#         #     # fallback for models expecting different order
#         #     res = model.train_model(
#         #         x_tr, z_tr, y_tr,
#         #         record_loss=True,
#         #         x_val=x_va, z_val=z_va, y_val=y_va
#         #     )

#         # extract train & val loss
#         train_losses, val_losses = res[0], res[1]

#         # plot
#         os.makedirs("img", exist_ok=True)
#         plt.figure(figsize=(8,6))
#         plt.plot(train_losses, label="Train Loss")
#         plt.plot(val_losses,   label="Val   Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.title(f"{method_name} Run {run}")
#         plt.legend()
#         plt.savefig(f"img/{method_name}_run{run}.png")
#         plt.close()

#         # evaluate on test
#         model.eval()
#         with torch.no_grad():
#             preds = model.forward(test_X, test_Z)[0]
#         auc = roc_auc_score(test_y.cpu().numpy(), preds.cpu().numpy())
#         metrics_list.append({'auc': auc})
#         print(f"{method_name} Run {run} AUC: {auc:.4f}")

#     return metrics_list

def grid_search_multiview(model_class, param_grid, X, Z, y, cv=3):
    """
    Perform grid search for multi-view models using ROC AUC on the main task.

    Returns:
        best_params: dict of hyperparameters with highest average AUC
        best_auc: float, corresponding AUC
    """
    best_auc = -np.inf
    best_params = None
    keys = list(param_grid.keys())

    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        aucs = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        for tr_idx, va_idx in kf.split(X):
            x_tr, z_tr, y_tr = X[tr_idx], Z[tr_idx], y[tr_idx]
            x_va, z_va, y_va = X[va_idx], Z[va_idx], y[va_idx]

            # Instantiate model with dynamic signature
            sig = inspect.signature(model_class.__init__).parameters
            init_kwargs = {}
            if 'dim_x' in sig:
                init_kwargs['dim_x'] = X.shape[1]
            if 'dim_z' in sig:
                init_kwargs['dim_z'] = Z.shape[1]
            for k, v in params.items():
                if k in sig:
                    init_kwargs[k] = v
            model = model_class(**init_kwargs)

            # Pretrain auxiliary branch if available
            if hasattr(model, 'pretrain_aux'):
                model.pretrain_aux(z_tr, y_tr)
            # Main training
            model.train_model(x_tr, z_tr, y_tr, record_loss=False)

            # Validation AUC
            model.eval()
            with torch.no_grad():
                out = model.forward(x_va, z_va)[0]
            aucs.append(roc_auc_score(y_va.cpu().numpy(), out.cpu().numpy()))

        avg_auc = float(np.mean(aucs))
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_params = params
    return best_params, best_auc


# def run_experiments_multiview(
#     model_class,
#     method_name,
#     num_runs,
#     best_params,
#     train_X,
#     train_Z,
#     train_y,
#     test_X,
#     test_Z,
#     test_y
# ):
#     """
#     Run multiple random-sampled training runs for a multi-view model.

#     For each run:
#       - Balanced sampling of up to 250 examples per class
#       - 80/20 train/validation split
#       - Pretrain auxiliary if supported
#       - Train with recording losses
#       - Plot and save loss curves
#       - Evaluate on held-out test set

#     Returns:
#         metrics_list: list of dicts with 'auc' per run
#     """
#     metrics = []
#     # gather class indices
#     y_np = train_y.cpu().numpy()
#     idx_pos = np.where(y_np == 1)[0].tolist()
#     idx_neg = np.where(y_np == 0)[0].tolist()
#     test_pool_size = len(test_X) 

#     for run in range(1, num_runs + 1):
#         # balanced sampling
#         samp1 = random.sample(idx_pos, min(len(idx_pos), 250))
#         samp0 = random.sample(idx_neg, min(len(idx_neg), 250))
#         idx_train = samp1 + samp0

#         x_pool = train_X[idx_train]
#         z_pool = train_Z[idx_train]
#         y_pool = train_y[idx_train]

#         test_indices = random.sample(range(test_pool_size), 125)
#         x_test_run = test_X[test_indices]
#         y_test_run = test_y[test_indices]
#         z_test_run = test_Z[test_indices]

#         # train/val split
#         perm = torch.randperm(len(x_pool))
#         split = int(0.8 * len(x_pool))
#         tr_idx, va_idx = perm[:split], perm[split:]
#         x_tr, z_tr, y_tr = x_pool[tr_idx], z_pool[tr_idx], y_pool[tr_idx]
#         x_va, z_va, y_va = x_pool[va_idx], z_pool[va_idx], y_pool[va_idx]

#         # instantiate model
#         sig = inspect.signature(model_class.__init__).parameters
#         init_kwargs = {}
#         if 'dim_x' in sig:
#             init_kwargs['dim_x'] = train_X.shape[1]
#         if 'dim_z' in sig:
#             init_kwargs['dim_z'] = train_Z.shape[1]
#         for k, v in best_params.items():
#             if k in sig:
#                 init_kwargs[k] = v
#         model = model_class(**init_kwargs)

#         # pretrain auxiliary if available
#         if hasattr(model, 'pretrain_aux'):
#             model.pretrain_aux(z_tr, y_tr)

#         # train with loss recording
#         losses = model.train_model(x_tr, z_tr, y_tr, record_loss=True)
#         train_losses, val_losses = losses

#         # plot losses
#         filename = f"img/{method_name}_run{run}.png"
#         plot_loss_curve(train_losses, val_losses, filename)

#         # evaluate on test
#         # model.eval()
#         # with torch.no_grad():
#         #     pred = model.forward(test_X, test_Z)[0]
#         # auc = roc_auc_score(test_y.cpu().numpy(), pred.cpu().numpy())
#         # metrics.append({'auc': auc})
#         # print(f"{method_name} Run {run} AUC: {auc:.4f}")
#         metric = model.evaluate(x_test_run, y_test_run, z_test_run)
#         # metric = model.evaluate(test_X, test_y, test_Z)
#         metrics.append(metric)
#         print(metric)
#     return metrics

# def run_experiments_baseline(
#     model_class,
#     method_name,
#     num_runs,
#     best_params,
#     train_x_pool,
#     train_y_pool,
#     test_x_pool,
#     test_y_pool):
#     metrics_list = []
#     # compute class indices once
#     train_df = pd.DataFrame(train_x_pool.numpy())
#     train_df['Diabetes_binary'] = train_y_pool.numpy()
#     class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
#     class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
#     test_pool_size = len(test_x_pool)

#     # pre‑grab the __init__ signature
#     sig = inspect.signature(model_class.__init__).parameters

#     for run_idx in range(1, num_runs + 1):
#         # 1) Balanced sampling
#         sampled1 = random.sample(class1_indices, min(250, len(class1_indices)))
#         sampled0 = random.sample(class0_indices, min(250, len(class0_indices)))
#         train_idx = sampled1 + sampled0
#         x_train = train_x_pool[train_idx]
#         y_train = train_y_pool[train_idx]
#         # 2) Pull out a random test subset
#         test_idx = random.sample(range(test_pool_size), min(125, test_pool_size))
#         x_test = test_x_pool[test_idx]
#         y_test = test_y_pool[test_idx]
#         # 3) Instantiate the model dynamically:
#         init_kwargs = {}
#         if 'input_dim' in sig:
#             init_kwargs['input_dim'] = x_train.shape[1]
#         # feed only the keys that actually appear in its __init__
#         for k, v in best_params.items():
#             if k in sig:
#                 init_kwargs[k] = v
#         model = model_class(**init_kwargs)

#         # 4) Split train→val (80/20)
#         perm = np.random.permutation(len(x_train)); split=int(0.8*len(x_train))
#         tr, va = perm[:split], perm[split:]
#         x_tr, y_tr = x_train[tr], y_train[tr]
#         x_va, y_va = x_train[va], y_train[va]
#         train_losses, val_losses = model.train_model(
#             x_tr, y_tr,
#             x_val=x_va, y_val=y_va,
#             early_stopping_patience=10,
#             record_loss=True
#         )
#         # 5) Plot loss curves
#         os.makedirs("img", exist_ok=True)
#         fn = f"img/{method_name}_run{run_idx}.png"
#         plt.figure(figsize=(8,6))
#         plt.plot(train_losses, label="Train BCE Loss")
#         plt.plot(val_losses,   label="Val   BCE Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("BCE Loss")
#         plt.title(f"{method_name} run {run_idx}")
#         plt.legend()
#         plt.savefig(fn)
#         plt.close()

#         # 6) Evaluate
#         metrics = model.evaluate(x_test, y_test)
#         print(f"Run {run_idx} test metrics:", metrics)
#         metrics_list.append(metrics)
#     print("-----", metrics_list)
#     return metrics_list

# # def run_experiments(
# #     model_class,
# #     method_name,
# #     num_runs,
# #     best_params,
# #     train_x_pool,
# #     train_y_pool,
# #     test_x_pool,
# #     test_y_pool,
# #     is_multitask=False,
# #     train_aux1_pool=None,
# #     train_aux2_pool=None,
# #     train_aux3_pool=None,
# # ):
# #     metrics_list = []
# #     # compute class indices once
# #     train_df = pd.DataFrame(train_x_pool.numpy())
# #     train_df['Diabetes_binary'] = train_y_pool.numpy()
# #     class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
# #     class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
# #     test_pool_size = len(test_x_pool)

# #     # pre‑grab the __init__ signature
# #     sig = inspect.signature(model_class.__init__).parameters

# #     for run_idx in range(1, num_runs + 1):
# #         # 1) Balanced sampling
# #         sampled1 = random.sample(class1_indices, 250)
# #         print("class1_indices",sampled1)
# #         sampled0 = random.sample(class0_indices, 250)
# #         train_idx = sampled1 + sampled0
# #         x_train = train_x_pool[train_idx]
# #         y_train = train_y_pool[train_idx]

# #         aux1_train = train_aux1_pool[train_idx]
# #         aux2_train = train_aux2_pool[train_idx]
# #         aux3_train = train_aux3_pool[train_idx]

# #         # 2) Pull out a random test subset
# #         test_idx = random.sample(range(test_pool_size), min(125, test_pool_size))
# #         x_test = test_x_pool[test_idx]
# #         y_test = test_y_pool[test_idx]

# #         # 3) Instantiate the model dynamically:
# #         init_kwargs = {}
# #         init_kwargs['input_dim'] = x_train.shape[1]
# #         # feed only the keys that actually appear in its __init__
# #         for k, v in best_params.items():
# #             if k in sig:
# #                 init_kwargs[k] = v
# #         model = model_class(**init_kwargs)

# #         # 4) Split train→val (80/20)
# #         perm = np.random.permutation(len(x_train)); split=int(0.8*len(x_train))
# #         tr, va = perm[:split], perm[split:]
# #         x_tr, y_tr = x_train[tr], y_train[tr]
# #         x_va, y_va = x_train[va], y_train[va]
# #         if is_multitask:
# #             a1_tr, a2_tr, a3_tr = aux1_train[tr], aux2_train[tr], aux3_train[tr]
# #             a1_va, a2_va, a3_va = aux1_train[va], aux2_train[va], aux3_train[va]
# #             res = model.train_model(
# #                 x_tr, y_tr, a1_tr, a2_tr, a3_tr,
# #                 x_val=x_va, main_y_val=y_va,
# #                 early_stopping_patience=10,
# #                 record_loss=True
# #             )
# #         if isinstance(res, tuple) and len(res) == 3:
# #             _, train_losses, val_losses = res
# #         else:
# #             train_losses, val_losses = res

# #         # 5) Plot loss curves
# #         os.makedirs("img", exist_ok=True)
# #         fn = f"img/{method_name}_run{run_idx}.png"
# #         plt.figure(figsize=(8,6))
# #         plt.plot(train_losses, label="Train BCE Loss")
# #         plt.plot(val_losses,   label="Val   BCE Loss")
# #         plt.xlabel("Epoch")
# #         plt.ylabel("BCE Loss")
# #         plt.title(f"{method_name} run {run_idx}")
# #         plt.legend()
# #         plt.savefig(fn)
# #         plt.close()

# #         # 6) Evaluate
# #         metrics = model.evaluate(x_test, y_test)
# #         print(f"Run {run_idx} test metrics:", metrics)
# #         metrics_list.append(metrics)

# #     return metrics_list

# def run_experiments(model_class, method_name, num_runs, best_params,
#                     train_x_pool, train_y_pool, test_x_pool, test_y_pool,
#                     optimizer_fixed="adam", epochs_fixed=300,
#                     is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
#     metrics_list = []
#     class_counts = []
#     train_df = pd.DataFrame(train_x_pool.numpy())
#     train_df['Diabetes_binary'] = train_y_pool.numpy()
#     class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
#     class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
#     test_pool_size = len(test_x_pool)
    
#     lr_val = best_params.get('lr', 0.01)  # default fallback
#     hidden_dim_val = best_params.get('hidden_dim', 64)
#     num_layers_val = best_params.get('num_layers', 1)
#     lambda_aux_val = best_params.get('lambda_aux', 0.3)
#     # lr_val = best_params['lr']
#     # hidden_dim_val = best_params['hidden_dim']
#     # num_layers_val = best_params['num_layers']
#     # lambda_aux_val = best_params['lambda_aux'] if 'lambda_aux' in best_params else 0.3
#     for run_idx in range(num_runs):
#         sampled1 = random.sample(class1_indices, 250)
#         # print(sampled1)
#         sampled0 = random.sample(class0_indices, 250)
#         train_idx = sampled1 + sampled0
#         x_train_run = train_x_pool[train_idx]
#         y_train_run = train_y_pool[train_idx]
#         n_pos = int((y_train_run == 1).sum().item())
#         n_neg = int((y_train_run == 0).sum().item())
#         class_counts.append({
#             'run': run_idx+1,
#             'n_neg': n_neg,
#             'n_pos': n_pos
#         })
#         print(f"Run {run_idx+1}: negative={n_neg}, positive={n_pos}")

#         if is_multitask:
#             aux1_train_run = train_aux1_pool[train_idx]
#             aux2_train_run = train_aux2_pool[train_idx]
#             aux3_train_run = train_aux3_pool[train_idx]
#         # ─────────────────────────────────────────────────────────        
#         # Randomly select 125 samples from the test pool
#         test_indices = random.sample(range(test_pool_size), 125)
#         x_test_run = test_x_pool[test_indices]
#         y_test_run = test_y_pool[test_indices]
#         print(model_class.__name__)
#         if model_class.__name__ == "MultiTaskLogisticRegression":
#             model = model_class(
#                 input_dim=x_train_run.shape[1],
#                 lr=lr_val,
#                 epochs=epochs_fixed,
#                 optimizer_type=optimizer_fixed
#             )
#         elif model_class.__name__ == "MultiTaskNN_PretrainFinetuneExtended":
#             model = model_class(
#                 input_dim=x_train_run.shape[1],
#                 hidden_dim=best_params['hidden_dim'],
#                 num_layers=best_params['num_layers'],
#                 lambda_aux=best_params['lambda_aux'],
#                 lr_pre=best_params['lr_pre'],
#                 lr_fine=best_params['lr_fine'],
#                 pre_epochs=best_params['pre_epochs'],
#                 fine_epochs=best_params['fine_epochs']
#             )
#         elif model_class.__name__ == "MultiTaskNN_Decoupled":
#             model = model_class(
#                 input_dim=x_train_run.shape[1],
#                 hidden_dim=best_params['hidden_dim'],
#                 num_layers=best_params['num_layers'],
#                 lambda_aux=best_params['lambda_aux'],
#                 lr=best_params['lr'],
#                 pre_epochs=best_params.get('pre_epochs', 100),
#                 main_epochs=best_params.get('fine_epochs', 100)
#             )
#         else:  # MultiTaskNN
#             model = model_class(
#                 input_dim=x_train_run.shape[1],
#                 hidden_dim=hidden_dim_val,
#                 num_layers=num_layers_val,
#                 optimizer_type=optimizer_fixed,
#                 lr=lr_val,
#                 epochs=epochs_fixed,
#                 lambda_aux=lambda_aux_val
#             )
#         # model = model_class(
#         #     input_dim=x_train_run.shape[1],
#         #     hidden_dim=hidden_dim_val,
#         #     num_layers=num_layers_val,
#         #     optimizer_type=optimizer_fixed,
#         #     lr=lr_val,
#         #     epochs=epochs_fixed,
#         #     lambda_aux=lambda_aux_val
#         #     )
        
#         # Split training run data into training and validation sets (80/20 split)
#         indices = list(range(len(x_train_run)))
#         random.shuffle(indices)
#         split = int(0.8 * len(indices))
#         train_idx, val_idx = indices[:split], indices[split:]
#         x_train_sub = x_train_run[train_idx]
#         y_train_sub = y_train_run[train_idx]
#         x_val_sub = x_train_run[val_idx]
#         y_val_sub = y_train_run[val_idx]
        
#         aux1_train_sub = aux1_train_run[train_idx]
#         aux2_train_sub = aux2_train_run[train_idx]
#         aux3_train_sub = aux3_train_run[train_idx]
#         aux1_val_sub = aux1_train_run[val_idx]
#         aux2_val_sub = aux2_train_run[val_idx]
#         aux3_val_sub = aux3_train_run[val_idx]
#         loss_data = model.train_model(x_train_sub, y_train_sub, aux1_train_sub, aux2_train_sub, aux3_train_sub,
#                                           x_val=x_val_sub, main_y_val=y_val_sub, aux1_val=aux1_val_sub, aux2_val=aux2_val_sub, aux3_val=aux3_val_sub,
#                                           early_stopping_patience=30, record_loss=True)
#         # loss_data = model.train_model(x_train_sub, y_train_sub,
#         #                                   x_val=x_val_sub, y_val=y_val_sub,
#         #                                   early_stopping_patience=30, record_loss=True)
#         if model.__class__.__name__ == "MultiTaskNN_PretrainFinetuneExtended":
#             _, train_losses, val_losses = loss_data
#         else:
#             train_losses, val_losses = loss_data

#         # train_losses, val_losses = loss_data
        
#         if not os.path.exists("img"):
#             os.makedirs("img")
#         plot_filename = os.path.join("img", f"MT+MLP_7B_wosplit{run_idx+1}.png")
#         plt.figure(figsize=(8,6))
#         plt.plot(train_losses, label="Train BCE Loss")
#         plt.plot(val_losses, label="Validation BCE Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("BCE Loss")
#         plt.title("Train vs Validation BCE Loss")
#         plt.legend()
#         plt.savefig(plot_filename)
#         plt.show()
        
#         # Evaluate model on test set
#         metrics = model.evaluate(x_test_run, y_test_run)
#         metrics_list.append(metrics)
#         print(metrics)
#     return metrics_list

# # def run_experiments(model_class, method_name, num_runs, best_params,
# #                     train_x_pool, train_y_pool, test_x_pool, test_y_pool,
# #                     is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
# #     metrics_list = []
# #     for run in range(1, num_runs + 1):
# #         idx1 = train_y_pool.nonzero(as_tuple=True)[0]
# #         idx0 = (train_y_pool == 0).nonzero(as_tuple=True)[0]
# #         sampled_idx = torch.cat([
# #             idx1[torch.randperm(len(idx1))[:250]],
# #             idx0[torch.randperm(len(idx0))[:250]]
# #         ])
# #         x_train, y_train = train_x_pool[sampled_idx], train_y_pool[sampled_idx]

# #         if is_multitask:
# #             aux1 = train_aux1_pool[sampled_idx]
# #             aux2 = train_aux2_pool[sampled_idx]
# #             aux3 = train_aux3_pool[sampled_idx]

# #         split = int(0.8 * len(x_train))
# #         perm = torch.randperm(len(x_train))
# #         tr_idx, val_idx = perm[:split], perm[split:]

# #         sig = inspect.signature(model_class.__init__).parameters
# #         if 'input_dim' in sig:
# #             model = model_class(input_dim=x_train.shape[1], **best_params)
# #         else:
# #             model = model_class(**best_params)

# #         if is_multitask:
# #             train_losses, val_losses = model.train_model(
# #                 x_train[tr_idx], y_train[tr_idx],
# #                 aux1[tr_idx], aux2[tr_idx], aux3[tr_idx],
# #                 x_val=x_train[val_idx], main_y_val=y_train[val_idx],
# #                 aux1_val=aux1[val_idx], aux2_val=aux2[val_idx], aux3_val=aux3[val_idx],
# #                 early_stopping_patience=10, record_loss=True
# #             )
# #         else:
# #             train_losses, val_losses = model.train_model(
# #                 x_train[tr_idx], y_train[tr_idx],
# #                 x_val=x_train[val_idx], y_val=y_train[val_idx],
# #                 early_stopping_patience=10, record_loss=True
# #             )

# #         plot_loss_curve(train_losses, val_losses, f"img/{method_name}_run{run}.png")

# #         metrics = model.evaluate(test_x_pool, test_y_pool)
# #         print(f"Run {run}: {metrics}")
# #         metrics_list.append(metrics)

# #     return metrics_list

# def run_experiments_direct(
#     model_class,
#     best_params,
#     train_x, train_y, train_z,
#     test_x, test_y,
#     num_runs=10,
#     sample_size_per_class=250,
#     early_stopping=10
# ):
#     """
#     Execute multiple runs for a direct pattern model using best hyperparameters.

#     Args:
#         model_class: class, same as for grid_search_direct.
#         best_params: dict, hyperparameters from grid search.
#         train_x, train_y, train_z: torch.Tensor pools for training (N_train, D), (N_train,), (N_train,).
#         test_x, test_y: torch.Tensor for final evaluation.
#         num_runs: int, number of random balanced sampling experiments.
#         sample_size_per_class: int, how many examples per class to sample.
#         early_stopping: int, patience passed to train_model.

#     Returns:
#         metrics_list: list of dicts from model.evaluate() across runs.
#     """
#     metrics_list = []
#     # Identify indices for each class
#     train_df = pd.DataFrame(train_x.numpy())
#     train_df['Diabetes_binary'] = train_y.numpy()
#     class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
#     class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
#     test_pool_size = len(test_x)
#     for run_idx in range(num_runs):
#         # Balanced sampling
#         sampled1 = random.sample(class1_indices, 250)
#         sampled0 = random.sample(class0_indices, 250)
#         train_idx = sampled1 + sampled0
#         x_sel, y_sel, z_sel = train_x[train_idx], train_y[train_idx], train_z[train_idx]
#         # Split into train/val
#         perm = np.random.permutation(len(train_idx)); split=int(0.8*len(train_idx))
#         tr_i, va_i = perm[:split], perm[split:]
#         # print(tr_i)
#         test_indices = random.sample(range(test_pool_size), 125)

#         x_test_run = test_x[test_indices]
#         y_test_run = test_y[test_indices]

#         model = model_class(input_dim=train_x.shape[1], **best_params)
#         train_losses, val_losses = model.train_model(
#             x_sel[tr_i], y_sel[tr_i], z_sel[tr_i],
#             x_val=x_sel[va_i], y_val=y_sel[va_i], z_val=z_sel[va_i],
#             early_stopping=early_stopping,
#             record_loss=True
#         )
#         plot_loss_curve(train_losses, val_losses, f"img/{model_class}_run{run_idx}.png")

#         # Evaluate on test set
#         metrics = model.evaluate(x_test_run, y_test_run)
#         metrics_list.append(metrics)
#         print(f"Run {run_idx}: {metrics}")

#     return metrics_list

def run_experiments_multiview(
    model_class,
    method_name,
    num_runs,
    best_params,
    train_X,
    train_Z,
    train_y,
    test_X,
    test_Z,
    test_y
):
    """
    Run multiple random-sampled training runs for a multi-view model.

    For each run:
      - Balanced sampling of up to 250 examples per class
      - 80/20 train/validation split
      - Pretrain auxiliary if supported
      - Train with recording losses
      - Plot and save loss curves
      - Evaluate on held-out test set

    Returns:
        metrics_list: list of dicts with 'auc' per run
    """
    metrics = []
    # gather class indices
    y_np = train_y.cpu().numpy()
    idx_pos = np.where(y_np == 1)[0].tolist()
    idx_neg = np.where(y_np == 0)[0].tolist()
    test_pool_size = len(test_X) 

    for run in range(1, num_runs + 1):
        # balanced sampling
        samp1 = random.sample(idx_pos, min(len(idx_pos), 250))
        samp0 = random.sample(idx_neg, min(len(idx_neg), 250))
        idx_train = samp1 + samp0

        x_pool = train_X[idx_train]
        z_pool = train_Z[idx_train]
        y_pool = train_y[idx_train]

        test_indices = random.sample(range(test_pool_size), 125)
        x_test_run = test_X[test_indices]
        y_test_run = test_y[test_indices]
        z_test_run = test_Z[test_indices]

        # train/val split
        perm = torch.randperm(len(x_pool))
        split = int(0.8 * len(x_pool))
        tr_idx, va_idx = perm[:split], perm[split:]
        x_tr, z_tr, y_tr = x_pool[tr_idx], z_pool[tr_idx], y_pool[tr_idx]
        x_va, z_va, y_va = x_pool[va_idx], z_pool[va_idx], y_pool[va_idx]

        # instantiate model
        sig = inspect.signature(model_class.__init__).parameters
        init_kwargs = {}
        if 'dim_x' in sig:
            init_kwargs['dim_x'] = train_X.shape[1]
        if 'dim_z' in sig:
            init_kwargs['dim_z'] = train_Z.shape[1]
        for k, v in best_params.items():
            if k in sig:
                init_kwargs[k] = v
        model = model_class(**init_kwargs)

        # pretrain auxiliary if available
        if hasattr(model, 'pretrain_aux'):
            model.pretrain_aux(z_tr, y_tr)

        # train with loss recording
        losses = model.train_model(x_tr, z_tr, y_tr, record_loss=True)
        train_losses, val_losses = losses

        # plot losses
        filename = f"img/{method_name}_run{run}.png"
        plot_loss_curve(train_losses, val_losses, filename)

        # evaluate on test
        # model.eval()
        # with torch.no_grad():
        #     pred = model.forward(test_X, test_Z)[0]
        # auc = roc_auc_score(test_y.cpu().numpy(), pred.cpu().numpy())
        # metrics.append({'auc': auc})
        # print(f"{method_name} Run {run} AUC: {auc:.4f}")
        metric = model.evaluate(x_test_run, y_test_run, z_test_run)
        # metric = model.evaluate(test_X, test_y, test_Z)
        metrics.append(metric)
        print(metric)
    return metrics

def run_experiments_baseline(
    model_class,
    method_name,
    num_runs,
    best_params,
    train_x_pool,
    train_y_pool,
    test_x_pool,
    test_y_pool):
    metrics_list = []
    # compute class indices once
    train_df = pd.DataFrame(train_x_pool.numpy())
    train_df['Diabetes_binary'] = train_y_pool.numpy()
    class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
    class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
    test_pool_size = len(test_x_pool)

    # pre‑grab the __init__ signature
    sig = inspect.signature(model_class.__init__).parameters

    for run_idx in range(1, num_runs + 1):
        # 1) Balanced sampling
        sampled1 = random.sample(class1_indices, min(250, len(class1_indices)))
        sampled0 = random.sample(class0_indices, min(250, len(class0_indices)))
        train_idx = sampled1 + sampled0
        x_train = train_x_pool[train_idx]
        y_train = train_y_pool[train_idx]
        # 2) Pull out a random test subset
        test_idx = random.sample(range(test_pool_size), min(125, test_pool_size))
        x_test = test_x_pool[test_idx]
        y_test = test_y_pool[test_idx]
        # 3) Instantiate the model dynamically:
        init_kwargs = {}
        if 'input_dim' in sig:
            init_kwargs['input_dim'] = x_train.shape[1]
        # feed only the keys that actually appear in its __init__
        for k, v in best_params.items():
            if k in sig:
                init_kwargs[k] = v
        model = model_class(**init_kwargs)

        # 4) Split train→val (80/20)
        perm = np.random.permutation(len(x_train)); split=int(0.8*len(x_train))
        tr, va = perm[:split], perm[split:]
        x_tr, y_tr = x_train[tr], y_train[tr]
        x_va, y_va = x_train[va], y_train[va]
        train_losses, val_losses = model.train_model(
            x_tr, y_tr,
            x_val=x_va, y_val=y_va,
            early_stopping_patience=10,
            record_loss=True
        )
        # 5) Plot loss curves
        os.makedirs("img", exist_ok=True)
        fn = f"img/{method_name}_run{run_idx}.png"
        plt.figure(figsize=(8,6))
        plt.plot(train_losses, label="Train BCE Loss")
        plt.plot(val_losses,   label="Val   BCE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title(f"{method_name} run {run_idx}")
        plt.legend()
        plt.savefig(fn)
        plt.close()

        # 6) Evaluate
        metrics = model.evaluate(x_test, y_test)
        print(f"Run {run_idx} test metrics:", metrics)
        metrics_list.append(metrics)
    print("-----", metrics_list)
    return metrics_list

def run_experiments(model_class, method_name, num_runs, best_params,
                    train_x_pool, train_y_pool, test_x_pool, test_y_pool,
                    optimizer_fixed="adam", epochs_fixed=300,
                    is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
    metrics_list = []
    class_counts = []
    train_df = pd.DataFrame(train_x_pool.numpy())
    train_df['Diabetes_binary'] = train_y_pool.numpy()
    class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
    class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
    test_pool_size = len(test_x_pool)
    
    lr_val = best_params.get('lr', 0.01)  # default fallback
    hidden_dim_val = best_params.get('hidden_dim', 64)
    num_layers_val = best_params.get('num_layers', 1)
    lambda_aux_val = best_params.get('lambda_aux', 0.3)
    # lr_val = best_params['lr']
    # hidden_dim_val = best_params['hidden_dim']
    # num_layers_val = best_params['num_layers']
    # lambda_aux_val = best_params['lambda_aux'] if 'lambda_aux' in best_params else 0.3
    for run_idx in range(num_runs):
        sampled1 = random.sample(class1_indices, 250)
        sampled0 = random.sample(class0_indices, 250)
        train_idx = sampled1 + sampled0
        x_train_run = train_x_pool[train_idx]
        y_train_run = train_y_pool[train_idx]
        n_pos = int((y_train_run == 1).sum().item())
        n_neg = int((y_train_run == 0).sum().item())
        class_counts.append({
            'run': run_idx+1,
            'n_neg': n_neg,
            'n_pos': n_pos
        })
        print(f"Run {run_idx+1}: negative={n_neg}, positive={n_pos}")

        if is_multitask:
            aux1_train_run = train_aux1_pool[train_idx]
            aux2_train_run = train_aux2_pool[train_idx]
            aux3_train_run = train_aux3_pool[train_idx]
        # ─────────────────────────────────────────────────────────        
        # Randomly select 125 samples from the test pool
        test_indices = random.sample(range(test_pool_size), 125)
        x_test_run = test_x_pool[test_indices]
        y_test_run = test_y_pool[test_indices]
        if model_class.__name__ == "MultiTaskLogisticRegression":
            model = model_class(
                input_dim=x_train_run.shape[1],
                lr=lr_val,
                epochs=epochs_fixed,
                optimizer_type=optimizer_fixed
            )
        elif model_class.__name__ == "MultiTaskNN_PretrainFinetuneExtended":
            model = model_class(
                input_dim=x_train_run.shape[1],
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers'],
                lambda_aux=best_params['lambda_aux'],
                lr_pre=best_params['lr_pre'],
                lr_fine=best_params['lr_fine'],
                pre_epochs=best_params['pre_epochs'],
                fine_epochs=best_params['fine_epochs']
            )
        elif model_class.__name__ == "MultiTaskNN_Decoupled":
            model = model_class(
                input_dim=x_train_run.shape[1],
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers'],
                lambda_aux=best_params['lambda_aux'],
                lr=best_params['lr'],
                pre_epochs=best_params.get('pre_epochs', 100),
                main_epochs=best_params.get('fine_epochs', 100)
            )
        else:  # MultiTaskNN
            model = model_class(
                input_dim=x_train_run.shape[1],
                hidden_dim=hidden_dim_val,
                num_layers=num_layers_val,
                optimizer_type=optimizer_fixed,
                lr=lr_val,
                epochs=epochs_fixed,
                lambda_aux=lambda_aux_val
            )
        # model = model_class(
        #     input_dim=x_train_run.shape[1],
        #     hidden_dim=hidden_dim_val,
        #     num_layers=num_layers_val,
        #     optimizer_type=optimizer_fixed,
        #     lr=lr_val,
        #     epochs=epochs_fixed,
        #     lambda_aux=lambda_aux_val
        #     )
        
        # Split training run data into training and validation sets (80/20 split)
        indices = list(range(len(x_train_run)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]
        x_train_sub = x_train_run[train_idx]
        y_train_sub = y_train_run[train_idx]
        x_val_sub = x_train_run[val_idx]
        y_val_sub = y_train_run[val_idx]
        
        aux1_train_sub = aux1_train_run[train_idx]
        aux2_train_sub = aux2_train_run[train_idx]
        aux3_train_sub = aux3_train_run[train_idx]
        aux1_val_sub = aux1_train_run[val_idx]
        aux2_val_sub = aux2_train_run[val_idx]
        aux3_val_sub = aux3_train_run[val_idx]
        loss_data = model.train_model(x_train_sub, y_train_sub, aux1_train_sub, aux2_train_sub, aux3_train_sub,
                                          x_val=x_val_sub, main_y_val=y_val_sub, aux1_val=aux1_val_sub, aux2_val=aux2_val_sub, aux3_val=aux3_val_sub,
                                          early_stopping_patience=30, record_loss=True)
        # loss_data = model.train_model(x_train_sub, y_train_sub,
        #                                   x_val=x_val_sub, y_val=y_val_sub,
        #                                   early_stopping_patience=30, record_loss=True)
        if model.__class__.__name__ == "MultiTaskNN_PretrainFinetuneExtended":
            _, train_losses, val_losses = loss_data
        else:
            train_losses, val_losses = loss_data

        # train_losses, val_losses = loss_data
        
        if not os.path.exists("img"):
            os.makedirs("img")
        plot_filename = os.path.join("img", f"MT+MLP_7B_wosplit{run_idx+1}.png")
        plt.figure(figsize=(8,6))
        plt.plot(train_losses, label="Train BCE Loss")
        plt.plot(val_losses, label="Validation BCE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("Train vs Validation BCE Loss")
        plt.legend()
        plt.savefig(plot_filename)
        plt.show()
        
        # Evaluate model on test set
        metrics = model.evaluate(x_test_run, y_test_run)
        metrics_list.append(metrics)
    return metrics_list

# def run_experiments(model_class, method_name, num_runs, best_params,
#                     train_x_pool, train_y_pool, test_x_pool, test_y_pool,
#                     is_multitask=False, train_aux1_pool=None, train_aux2_pool=None, train_aux3_pool=None):
#     metrics_list = []
#     for run in range(1, num_runs + 1):
#         idx1 = train_y_pool.nonzero(as_tuple=True)[0]
#         idx0 = (train_y_pool == 0).nonzero(as_tuple=True)[0]
#         sampled_idx = torch.cat([
#             idx1[torch.randperm(len(idx1))[:250]],
#             idx0[torch.randperm(len(idx0))[:250]]
#         ])
#         x_train, y_train = train_x_pool[sampled_idx], train_y_pool[sampled_idx]

#         if is_multitask:
#             aux1 = train_aux1_pool[sampled_idx]
#             aux2 = train_aux2_pool[sampled_idx]
#             aux3 = train_aux3_pool[sampled_idx]

#         split = int(0.8 * len(x_train))
#         perm = torch.randperm(len(x_train))
#         tr_idx, val_idx = perm[:split], perm[split:]

#         sig = inspect.signature(model_class.__init__).parameters
#         if 'input_dim' in sig:
#             model = model_class(input_dim=x_train.shape[1], **best_params)
#         else:
#             model = model_class(**best_params)

#         if is_multitask:
#             train_losses, val_losses = model.train_model(
#                 x_train[tr_idx], y_train[tr_idx],
#                 aux1[tr_idx], aux2[tr_idx], aux3[tr_idx],
#                 x_val=x_train[val_idx], main_y_val=y_train[val_idx],
#                 aux1_val=aux1[val_idx], aux2_val=aux2[val_idx], aux3_val=aux3[val_idx],
#                 early_stopping_patience=10, record_loss=True
#             )
#         else:
#             train_losses, val_losses = model.train_model(
#                 x_train[tr_idx], y_train[tr_idx],
#                 x_val=x_train[val_idx], y_val=y_train[val_idx],
#                 early_stopping_patience=10, record_loss=True
#             )

#         plot_loss_curve(train_losses, val_losses, f"img/{method_name}_run{run}.png")

#         metrics = model.evaluate(test_x_pool, test_y_pool)
#         print(f"Run {run}: {metrics}")
#         metrics_list.append(metrics)

#     return metrics_list

def run_experiments_direct(
    model_class,
    best_params,
    train_x, train_y, train_z,
    test_x, test_y,
    num_runs=10,
    sample_size_per_class=250,
    early_stopping=10
):
    """
    Execute multiple runs for a direct pattern model using best hyperparameters.

    Args:
        model_class: class, same as for grid_search_direct.
        best_params: dict, hyperparameters from grid search.
        train_x, train_y, train_z: torch.Tensor pools for training (N_train, D), (N_train,), (N_train,).
        test_x, test_y: torch.Tensor for final evaluation.
        num_runs: int, number of random balanced sampling experiments.
        sample_size_per_class: int, how many examples per class to sample.
        early_stopping: int, patience passed to train_model.

    Returns:
        metrics_list: list of dicts from model.evaluate() across runs.
    """
    metrics_list = []
    # Identify indices for each class
    train_df = pd.DataFrame(train_x.numpy())
    train_df['Diabetes_binary'] = train_y.numpy()
    class1_indices = train_df[train_df['Diabetes_binary'] == 1].index.tolist()
    class0_indices = train_df[train_df['Diabetes_binary'] == 0].index.tolist()
    test_pool_size = len(test_x)
    for run_idx in range(num_runs):
        # Balanced sampling
        sampled1 = random.sample(class1_indices, 250)
        sampled0 = random.sample(class0_indices, 250)
        train_idx = sampled1 + sampled0
        x_sel, y_sel, z_sel = train_x[train_idx], train_y[train_idx], train_z[train_idx]
        # Split into train/val
        perm = np.random.permutation(len(train_idx)); split=int(0.8*len(train_idx))
        tr_i, va_i = perm[:split], perm[split:]
        # print(tr_i)
        test_indices = random.sample(range(test_pool_size), 125)

        x_test_run = test_x[test_indices]
        y_test_run = test_y[test_indices]

        model = model_class(input_dim=train_x.shape[1], **best_params)
        train_losses, val_losses = model.train_model(
            x_sel[tr_i], y_sel[tr_i], z_sel[tr_i],
            x_val=x_sel[va_i], y_val=y_sel[va_i], z_val=z_sel[va_i],
            early_stopping=early_stopping,
            record_loss=True
        )
        plot_loss_curve(train_losses, val_losses, f"img/{model_class}_run{run_idx}.png")

        # Evaluate on test set
        metrics = model.evaluate(x_test_run, y_test_run)
        metrics_list.append(metrics)
        print(f"Run {run_idx}: {metrics}")

    return metrics_list


def run_experiments_pairwise(model_cls, params, train_set, test_set, runs=10, sample_size_per_class=250):
    Xi, Xj, S, yi, yj = train_set
    test_X, test_y = test_set
    metrics_list = []
    idx1=(yi[:,0]==1).nonzero(as_tuple=True)[0].cpu().numpy()
    idx0=(yi[:,0]==0).nonzero(as_tuple=True)[0].cpu().numpy()
    for run in range(runs):
        sel1=np.random.choice(idx1, min(len(idx1),sample_size_per_class), replace=False)
        sel0=np.random.choice(idx0, min(len(idx0),sample_size_per_class), replace=False)
        sel=np.concatenate([sel1,sel0]); np.random.shuffle(sel)
        tr,va=sel[:int(0.8*len(sel))], sel[int(0.8*len(sel)):]
        init={k:params[k] for k in ['hidden_dim','lr','epochs'] if k in params}
        model=model_cls(input_dim=Xi.shape[1], **init)
        losses=model.train_model(
            Xi[tr],Xj[tr],S[tr],yi[tr],yj[tr],
            Xi[va],Xj[va],S[va],yi[va],yj[va],
            margin=params.get('margin',1.0),
            lambda_w=params.get('lambda_w',0.5),
            gamma=params.get('gamma',1.0),
            record_loss=True
        )
        tr_l,va_l=losses
        plot_loss_curve(tr_l,va_l,f"img/{model_cls.__name__}_run{run}.png")
        metrics = model.evaluate(
            Xi[va], Xj[va], S[va], yi[va], yj[va],
            margin=params["margin"],
            lambda_w=params["lambda_w"],
            gamma=params["gamma"]
        )
        metrics_list.append(metrics)
        print(f"Run {run}: {metrics}")

    return metrics_list

def aggregate_metrics(metrics_list):
    agg = {}
    for k in metrics_list[0].keys():
        vals = [m[k] for m in metrics_list]
        agg[k] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
    return agg


def write_results_csv(filename, method, metrics, params):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            method,
            metrics.get('auc', ''),
            metrics.get('overall_acc', ''),
            metrics.get('a0', ''), metrics.get('p0', ''), metrics.get('r0', ''), metrics.get('f0', ''),
            metrics.get('a1', ''), metrics.get('p1', ''), metrics.get('r1', ''), metrics.get('f1', ''),
            metrics.get('bce_loss', ''),
            params
        ])


def print_aggregated_results(method, metrics, params):
    print(f"\n=== Final Aggregated Results ===")
    print(f"Method: {method}")
    print(f"AUC: {metrics.get('auc','')}")
    print(f"Overall Accuracy: {metrics.get('overall_acc','')}")
    print(f"Class0 -> Accuracy: {metrics.get('a0','')}, Precision: {metrics.get('p0','')}, Recall: {metrics.get('r0','')}, F1: {metrics.get('f0','')}")
    print(f"Class1 -> Accuracy: {metrics.get('a1','')}, Precision: {metrics.get('p1','')}, Recall: {metrics.get('r1','')}, F1: {metrics.get('f1','')}")
    print(f"Test BCE Loss: {metrics.get('bce_loss','')}")
    print(f"Best Parameter Set: {params}")
    print("---------------------------------------------------")
