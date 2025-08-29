import os
import csv
import itertools
import random
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from ucimlrepo import fetch_ucirepo
from tqdm import tqdm

# =============================================================================
# 1. Set Random Seed and Device
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 2. Plot Loss Curves Function
# =============================================================================
def plot_loss_curves(train_losses, val_losses, filename, title):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# =============================================================================
# 3. Data Preprocessing Class for Multi-view + Multi-task Setting
#
# 이 클래스는 UCI 데이터와 보조 CSV 파일 (예: prompting/augmented_data.csv)을 불러와서:
#   - View1: 주요 feature (x)
#   - View2: side information (z)
#   - Main target: Diabetes_binary (y_main)
#   - Auxiliary targets: health_1_10 (aux1), diabetes_risk_score (aux2), has_diabetes (aux3)
# 데이터를 스케일링한 후 torch.Tensor로 변환합니다.
# =============================================================================
class MultiViewDatasetPreprocessor:
    def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features  # DataFrame
        self.y_original = self.original_data.data.targets      # Series
        self.side_info_path = side_info_path
        self.scaler_view1 = StandardScaler()
        self.scaler_view2 = StandardScaler()
    
    def preprocess(self):
        # Define columns for view1 and view2.
        view1_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        # View2: side information
        view2_cols = ['predict_hba1c', 'predict_cholesterol', 'systolic_bp', 'diastolic_bp', 'exercise_freq', 'hi_sugar_freq', 'employment_status']
        target_col = 'Diabetes_binary'
        # Auxiliary targets
        aux_cols = ['health_1_10', 'diabetes_risk_score', 'has_diabetes']
        
        # Load augmented CSV file
        augmented_df = pd.read_csv(self.side_info_path).copy()
        selected_cols = view1_cols + view2_cols + [target_col] + aux_cols
        augmented_df = augmented_df[selected_cols]
        
        # Fill missing values for auxiliary targets
        augmented_df['has_diabetes'] = augmented_df['has_diabetes'].fillna(0)
        augmented_df['health_1_10'] = augmented_df['health_1_10'].fillna(augmented_df['health_1_10'].median())
        augmented_df['diabetes_risk_score'] = augmented_df['diabetes_risk_score'].fillna(augmented_df['diabetes_risk_score'].median())
        
        # Create a balanced grid for hyperparameter search
        pos_idx = augmented_df[augmented_df[target_col] == 1].index.tolist()
        neg_idx = augmented_df[augmented_df[target_col] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_df = pd.concat([
            augmented_df.loc[random.sample(pos_idx, n_pos)],
            augmented_df.loc[random.sample(neg_idx, n_neg)]
        ])
        
        # Training pool: remaining data
        train_pool_df = augmented_df.drop(index=grid_df.index)
        
        # Test pool: from original UCI data (drop indices present in augmented_df)
        original_df = self.X_original.copy()
        original_df[target_col] = self.y_original
        test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')
        
        # Scale view1 (features)
        self.scaler_view1.fit(train_pool_df[view1_cols])
        train_pool_df[view1_cols] = self.scaler_view1.transform(train_pool_df[view1_cols])
        grid_df[view1_cols] = self.scaler_view1.transform(grid_df[view1_cols])
        test_pool_df[view1_cols] = self.scaler_view1.transform(test_pool_df[view1_cols])
        
        # Scale view2 (side info)
        self.scaler_view2.fit(train_pool_df[view2_cols])
        train_pool_df[view2_cols] = self.scaler_view2.transform(train_pool_df[view2_cols])
        grid_df[view2_cols] = self.scaler_view2.transform(grid_df[view2_cols])
        
        for col in view2_cols:
            # 중앙값으로 채우거나, 필요에 따라 다른 값으로 채울 수 있습니다.
            train_pool_df[col].fillna(train_pool_df[col].median(), inplace=True)
            grid_df[col].fillna(train_pool_df[col].median(), inplace=True)

        self.scaler_view2.fit(train_pool_df[view2_cols])
        train_pool_df[view2_cols] = self.scaler_view2.transform(train_pool_df[view2_cols])
        grid_df[view2_cols] = self.scaler_view2.transform(grid_df[view2_cols])
        # Helper: DataFrame to Tensor conversion
        def df_to_tensors(df):
            x = torch.tensor(df[view1_cols].values, dtype=torch.float32)
            z = torch.tensor(df[view2_cols].values, dtype=torch.float32)
            y_main = torch.tensor(df[target_col].values, dtype=torch.float32).view(-1, 1)
            aux1 = torch.tensor(df['health_1_10'].values, dtype=torch.float32).view(-1, 1)
            aux2 = torch.tensor(df['diabetes_risk_score'].values, dtype=torch.float32).view(-1, 1)
            aux3 = torch.tensor(df['has_diabetes'].values, dtype=torch.float32).view(-1, 1)
            return x, y_main, z, aux1, aux2, aux3
        
        grid_x, grid_y, grid_z, grid_aux1, grid_aux2, grid_aux3 = df_to_tensors(grid_df)
        train_x, train_y, train_z, train_aux1, train_aux2, train_aux3 = df_to_tensors(train_pool_df)
        test_x = torch.tensor(test_pool_df[view1_cols].values, dtype=torch.float32)
        test_y = torch.tensor(test_pool_df[target_col].values, dtype=torch.float32).view(-1, 1)
        
        return (grid_x, grid_y, grid_z, grid_aux1, grid_aux2, grid_aux3,
                train_x, train_y, train_z, train_aux1, train_aux2, train_aux3,
                test_x, test_y)

# =============================================================================
# 4. Simultaneous Multi-view + Multi-task Model with Shared Prediction Head
#
# 이 모델은 두 입력 경로를 동시에 사용하여 main target을 예측합니다.
#
# 경로 1: X → EncoderX (φ) → s → SharedHead (ψ) → 
#          [Main prediction (from s), Aux1, Aux2, Aux3]  
#
# 경로 2: Z → EncoderZ (β) → s′ → SharedHead (ψ) → 
#          [Main prediction (from s′)]  (보조 타스크는 사용하지 않음)
#
# 두 경로 모두 같은 예측기 ψ를 사용합니다 (weight sharing).
#
# Loss 구성:
#   L_total = BCE(main_from_x, y) + BCE(main_from_z, y)
#             + λ_aux * (MSE(aux1, aux1_y) + MSE(aux2, aux2_y) + BCE(aux3, aux3_y))
#             + λ_direct * MSE(s, s′)
# =============================================================================
class SimultaneousMultiViewMultiTaskNN(nn.Module):
    def __init__(self, input_size_x, input_size_z, hidden_dim=64, epochs=300, lr=0.001,
                 lambda_aux=0.3, lambda_direct=0.3):
        super(SimultaneousMultiViewMultiTaskNN, self).__init__()
        self.epochs = epochs
        self.lambda_aux = lambda_aux
        self.lambda_direct = lambda_direct
        
        # Encoder for view1 (X) → s
        self.encoder_x = nn.Sequential(
            nn.Linear(input_size_x, hidden_dim),
            nn.ReLU()
        )
        # Encoder for view2 (Z) → s′
        self.encoder_z = nn.Sequential(
            nn.Linear(input_size_z, hidden_dim),
            nn.ReLU()
        )
        # Shared prediction head ψ: 동일 weight로 s와 s′에 적용
        # 이 head는 main task와 보조 task 3가지를 동시에 예측
        # 출력 차원: 1 (main), 1 (aux1), 1 (aux2), 1 (aux3) → 총 4
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 순서대로: main, aux1, aux2, aux3
        )
        
        # Loss 함수
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x, z):
        # Path 1: X → s
        s = self.encoder_x(x)
        # Path 2: Z → s′
        s_prime = self.encoder_z(z)
        # Shared head 적용: 같은 head로 예측 (weight sharing)
        out_from_x = self.shared_head(s)
        out_from_z = self.shared_head(s_prime)
        # out_from_x: [main_x, aux1, aux2, aux3]
        # out_from_z: [main_z, aux1_z, aux2_z, aux3_z] → 여기선 main_z만 사용
        main_x = torch.sigmoid(out_from_x[:, 0])
        aux1 = out_from_x[:, 1]  # 회귀: health_1_10
        aux2 = out_from_x[:, 2]  # 회귀: diabetes_risk_score
        aux3 = torch.sigmoid(out_from_x[:, 3])  # 이진: has_diabetes
        main_z = torch.sigmoid(out_from_z[:, 0])
        return s, s_prime, main_x, main_z, aux1, aux2, aux3
    
    def compute_loss(self, x, z, main_y, aux1_y, aux2_y, aux3_y):
        s, s_prime, main_x, main_z, aux1, aux2, aux3 = self.forward(x, z)
        
        # 타깃 값 클램핑 (필요하면)
        main_y = torch.clamp(main_y, 0.0, 1.0)
        aux3_y = torch.clamp(aux3_y, 0.0, 1.0)
        
        loss_main_x = self.bce_loss(main_x, main_y.view(-1))
        loss_main_z = self.bce_loss(main_z, main_y.view(-1))
        loss_main = loss_main_x + loss_main_z
        loss_aux1 = self.mse_loss(aux1, aux1_y.view(-1))
        loss_aux2 = self.mse_loss(aux2, aux2_y.view(-1))
        loss_aux3 = self.bce_loss(aux3, aux3_y.view(-1))
        loss_aux = loss_aux1 + loss_aux2 + loss_aux3
        loss_direct = self.mse_loss(s, s_prime)
        total_loss = loss_main + self.lambda_aux * loss_aux + self.lambda_direct * loss_direct
        return total_loss, loss_main, loss_aux, loss_direct


    
    def train_model(self, x, z, main_y, aux1_y, aux2_y, aux3_y,
                    x_val=None, z_val=None, main_y_val=None, aux1_val=None, aux2_val=None, aux3_val=None,
                    early_stopping_patience=10, record_loss=False):
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.train()
            self.optimizer.zero_grad()
            total_loss, loss_main, loss_aux, loss_direct = self.compute_loss(x, z, main_y, aux1_y, aux2_y, aux3_y)
            total_loss.backward()
            self.optimizer.step()
            if record_loss:
                train_loss_history.append(total_loss.item())
            if x_val is not None:
                self.eval()
                with torch.no_grad():
                    val_total_loss, _, _, _ = self.compute_loss(x_val, z_val, main_y_val, aux1_val, aux2_val, aux3_val)
                if record_loss:
                    val_loss_history.append(val_total_loss.item())
                if val_total_loss.item() < best_val_loss:
                    best_val_loss = val_total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        if record_loss:
            return train_loss_history, val_loss_history
    
    def evaluate(self, x, z, y):
        self.eval()
        with torch.no_grad():
            _, _, main_x, main_z, aux1, aux2, aux3 = self.forward(x, z)
            # Average the main task predictions from both paths
            main_pred = (main_x + main_z) / 2
            preds = (main_pred >= 0.5).float()
            # Overall accuracy and per-class accuracy
            accuracy = (preds == y.view(-1)).sum().item() / len(y)
            y_np = y.cpu().numpy().flatten()
            preds_np = preds.cpu().numpy().flatten()
            acc_class0 = np.mean(preds_np[y_np==0] == 0) if np.sum(y_np==0) > 0 else 0.0
            acc_class1 = np.mean(preds_np[y_np==1] == 1) if np.sum(y_np==1) > 0 else 0.0
            try:
                auc = roc_auc_score(y.cpu().numpy(), main_pred.cpu().numpy())
            except Exception:
                auc = 0.0
            bce = self.bce_loss(main_pred, y.view(-1)).item()
            precision, recall, f1, _ = precision_recall_fscore_support(
                y.cpu().numpy(), preds.cpu().numpy(), labels=[0,1], zero_division=0
            )
        return {
            "accuracy": accuracy,
            "accuracy_class0": acc_class0,
            "accuracy_class1": acc_class1,
            "auc": auc,
            "bce_loss": bce,
            "p0": precision[0],
            "r0": recall[0],
            "f0": f1[0],
            "p1": precision[1],
            "r1": recall[1],
            "f1": f1[1]
        }

# =============================================================================
# 5. Grid Search Function for Simultaneous Multi-view + Multi-task Model
#
# Grid search를 통해 최적의 하이퍼파라미터 (lr, hidden_dim, num_layers, lambda_aux, lambda_direct)를 찾습니다.
# =============================================================================
def grid_search_cv(model_class, param_grid, X, y, z, aux1, aux2, aux3, cv=3, lr_default=0.001):
    best_auc = -1.0
    best_params = None
    param_keys = list(param_grid.keys())
    kf = KFold(n_splits=cv, shuffle=True, random_state=SEED)
    
    for combination in itertools.product(*(param_grid[k] for k in param_keys)):
        combi = dict(zip(param_keys, combination))
        aucs = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            z_train, z_val = z[train_idx], z[val_idx]
            aux1_train, aux1_val = aux1[train_idx], aux1[val_idx]
            aux2_train, aux2_val = aux2[train_idx], aux2[val_idx]
            aux3_train, aux3_val = aux3[train_idx], aux3[val_idx]
            
            model = model_class(
                input_size_x = X.shape[1],
                input_size_z = z.shape[1],
                hidden_dim = combi.get('hidden_dim', 64),
                epochs = 100,  # 짧은 학습으로 그리드 서치
                lr = combi.get('lr', lr_default),
                lambda_aux = combi.get('lambda_aux', 0.3),
                lambda_direct = combi.get('lambda_direct', 0.3)
            ).to(device)
            model.train_model(
                X_train.to(device), z_train.to(device), y_train.to(device),
                aux1_train.to(device), aux2_train.to(device), aux3_train.to(device),
                x_val=X_val.to(device), z_val=z_val.to(device), main_y_val=y_val.to(device),
                aux1_val=aux1_val.to(device), aux2_val=aux2_val.to(device), aux3_val=aux3_val.to(device),
                early_stopping_patience=10, record_loss=False
            )
            model.eval()
            with torch.no_grad():
                _, _, main_x_val, main_z_val, _, _, _ = model.forward(X_val.to(device), z_val.to(device))
                main_pred_val = (main_x_val + main_z_val) / 2
                try:
                    auc_val = roc_auc_score(y_val.cpu().numpy(), main_pred_val.cpu().numpy())
                except Exception:
                    auc_val = 0.0
            aucs.append(auc_val)
        avg_auc = np.mean(aucs)
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_params = combi
    return best_params, best_auc

# =============================================================================
# 6. Run Experiments Function (10 Iterations)
# =============================================================================
def run_experiments(model_class, method_name, num_runs, best_params,
                    train_x, train_y, train_z, train_aux1, train_aux2, train_aux3,
                    test_x, test_y, test_z):
    metrics_list = []
    # Balanced sampling from training pool based on main target
    train_df = pd.DataFrame(train_x.cpu().numpy())
    train_df['target'] = train_y.cpu().numpy()
    class1_indices = train_df[train_df['target'] == 1].index.tolist()
    class0_indices = train_df[train_df['target'] == 0].index.tolist()
    for run in range(num_runs):
        print(f"[{method_name}] Run {run+1}/{num_runs}")
        n1 = min(len(class1_indices), 250)
        n0 = min(len(class0_indices), 250)
        sampled_class1 = random.sample(class1_indices, n1)
        sampled_class0 = random.sample(class0_indices, n0)
        balanced_idx = sampled_class1 + sampled_class0
        x_train_run = train_x[balanced_idx]
        y_train_run = train_y[balanced_idx]
        z_train_run = train_z[balanced_idx]
        aux1_run = train_aux1[balanced_idx]
        aux2_run = train_aux2[balanced_idx]
        aux3_run = train_aux3[balanced_idx]
        
        # Split into train/validation (80/20)
        indices = list(range(len(x_train_run)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_idx = indices[:split]
        val_idx = indices[split:]
        
        x_train_sub = x_train_run[train_idx]
        y_train_sub = y_train_run[train_idx]
        z_train_sub = z_train_run[train_idx]
        aux1_train_sub = aux1_run[train_idx]
        aux2_train_sub = aux2_run[train_idx]
        aux3_train_sub = aux3_run[train_idx]
        
        x_val_sub = x_train_run[val_idx]
        y_val_sub = y_train_run[val_idx]
        z_val_sub = z_train_run[val_idx]
        aux1_val_sub = aux1_run[val_idx]
        aux2_val_sub = aux2_run[val_idx]
        aux3_val_sub = aux3_run[val_idx]
        
        model = model_class(
            input_size_x = x_train_sub.shape[1],
            input_size_z = train_z.shape[1],
            hidden_dim = best_params["hidden_dim"],
            epochs = 300,
            lr = best_params["lr"],
            lambda_aux = best_params["lambda_aux"],
            lambda_direct = best_params["lambda_direct"]
        ).to(device)
        
        model.train_model(x_train_sub.to(device), z_train_sub.to(device), y_train_sub.to(device),
                          aux1_train_sub.to(device), aux2_train_sub.to(device), aux3_train_sub.to(device),
                          x_val=x_val_sub.to(device), z_val=z_val_sub.to(device), main_y_val=y_val_sub.to(device),
                          aux1_val=aux1_val_sub.to(device), aux2_val=aux2_val_sub.to(device), aux3_val=aux3_val_sub.to(device),
                          early_stopping_patience=10, record_loss=True)
        
        metrics = model.evaluate(test_x.to(device), test_z.to(device), test_y.to(device))
        print(f"Run {run+1} Metrics: {metrics}")
        metrics_list.append(metrics)
    return metrics_list

# =============================================================================
# 7. Aggregation Function: Aggregate metrics as "mean ± std"
# =============================================================================
def aggregate_metrics(metrics_list):
    agg = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        agg[key] = f"{np.mean(values):.2f} ± {np.std(values):.2f}"
    return agg

# =============================================================================
# 8. Main Function: Data Preprocessing, Grid Search, and Experiment Runs
# =============================================================================
def main():
    CSV_FILENAME = "final_results.csv"
    NUM_RUNS = 10

    print("\n=== Preprocessing Multi-view + Multi-task Dataset ===")
    preproc = MultiViewDatasetPreprocessor(dataset_id=891, side_info_path='prompting/augmented_data_70B.csv')
    (grid_x, grid_y, grid_z, grid_aux1, grid_aux2, grid_aux3,
     train_x, train_y, train_z, train_aux1, train_aux2, train_aux3,
     test_x, test_y) = preproc.preprocess()

    # Grid search using the grid (balanced) data
    param_grid = {
        "lr": [0.001, 0.01],
        "hidden_dim": [64, 128, 256],
        "num_layers": [1,2,3,4],
        "lambda_aux": [0.01, 0.1, 0.3],
        "lambda_direct": [0.01, 0.1, 0.3]
    }
    print("\n[Grid Search] Starting grid search for best hyperparameters...")
    best_params, best_auc = grid_search_cv(SimultaneousMultiViewMultiTaskNN, param_grid, grid_x, grid_y, grid_z, 
                                             grid_aux1, grid_aux2, grid_aux3, cv=3, lr_default=0.001)
    print(f"Best hyperparameters: {best_params}, Best AUC: {best_auc:.4f}")

    # Move training and test tensors to device
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    train_z = train_z.to(device)
    train_aux1 = train_aux1.to(device)
    train_aux2 = train_aux2.to(device)
    train_aux3 = train_aux3.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    # If test view2 data is not available, we create zeros with same dimension as train_z
    test_z = torch.zeros((test_x.shape[0], train_z.shape[1]), dtype=torch.float32).to(device)

    print("\n=== Running Final Experiments (10 Iterations) ===")
    results = run_experiments(SimultaneousMultiViewMultiTaskNN, "SimulMultiViewMultiTaskNN", NUM_RUNS,
                              best_params, train_x, train_y, train_z, train_aux1, train_aux2, train_aux3,
                              test_x, test_y, test_z)
    agg_metrics = aggregate_metrics(results)
    best_hyperparams_str = ", ".join([f"{k}: {v}" for k, v in best_params.items()])

    print("\n=== Final Aggregated Results ===")
    # print(f"Method: {method}")
    print(f"  AUC: {agg_metrics.get('auc', '')}")
    print(f"  Overall Accuracy: {agg_metrics.get('accuracy', '')}")
    print(f"  Class0 Accuracy: {agg_metrics.get('accuracy_class0', '')}")
    print(f"  Class1 Accuracy: {agg_metrics.get('accuracy_class1', '')}")
    print(f"  BCE Loss: {agg_metrics.get('bce_loss', '')}")
    print(f"  Precision (class0): {agg_metrics.get('p0', '')}, Recall (class0): {agg_metrics.get('r0', '')}, F1 (class0): {agg_metrics.get('f0', '')}")
    print(f"  Precision (class1): {agg_metrics.get('p1', '')}, Recall (class1): {agg_metrics.get('r1', '')}, F1 (class1): {agg_metrics.get('f1', '')}")
    print(f"  Best Parameter Set: {best_hyperparams_str}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    main()
