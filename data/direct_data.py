# data/direct_data.py

import pandas as pd
import torch
import random
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

class DirectDataPreprocessor:
    """
    Preprocessor for all Direct‑pattern models.
    """
    def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
        self.original = fetch_ucirepo(id=dataset_id)
        self.X_orig = self.original.data.features
        self.y_orig = self.original.data.targets
        self.side_path = side_info_path

        self.cols = [
            'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
            'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
            'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
            'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'
        ]
        self.target = 'Diabetes_binary'
        self.side_info = 'predict_hba1c'
        self.scaler = StandardScaler()

    def preprocess(self):
        continuous_cols = ['BMI', self.side_info]
        # 1) load augmented CSV
        df = pd.read_csv(self.side_path)
        df = df[self.cols + [self.target, self.side_info]]
        df[self.target].fillna(0, inplace=True)
        df[self.side_info].fillna(df[self.side_info].median(), inplace=True)

        # 2) create balanced grid for hyperparameter search
        pos_idx = df[df['Diabetes_binary'] == 1].index.tolist()
        neg_idx = df[df['Diabetes_binary'] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_pos = df.loc[random.sample(pos_idx, n_pos)]
        grid_neg = df.loc[random.sample(neg_idx, n_neg)]
        grid_df = pd.concat([grid_pos, grid_neg])
        
        train_pool = df.drop(index = grid_df.index)
        # 3) build test pool from original UCI data (no side_info column)
        orig_df = self.X_orig.copy()
        orig_df[self.target] = self.y_orig
        test_pool = orig_df.drop(df.index, errors='ignore')

        # 4) fit scaler on train_pool (BMI + side_info) and transform grid & train_pool
        self.scaler.fit(train_pool[['BMI', self.side_info]])
        grid_df[['BMI', self.side_info]]       = self.scaler.transform(grid_df[['BMI', self.side_info]])
        train_pool[['BMI', self.side_info]]  = self.scaler.transform(train_pool[['BMI', self.side_info]])

        # 5) transform only BMI in test_pool using the same scaler's BMI parameters
        bmi_mean  = self.scaler.mean_[0]
        bmi_scale = self.scaler.scale_[0]
        test_pool['BMI'] = (test_pool['BMI'] - bmi_mean) / bmi_scale

        # 6) to PyTorch tensors
        def to_tensor(d, multi=False):
            x = torch.tensor(d[self.cols].values, dtype=torch.float)
            y = torch.tensor(d[self.target].values, dtype=torch.float).view(-1,1)
            if multi:
                z = torch.tensor(d[self.side_info].values, dtype=torch.float).view(-1,1)
                return x, y, z
            else:
                return x, torch.tensor(d[self.target].values, dtype=torch.float)

        grid_x, grid_y, grid_z = to_tensor(grid_df, multi=True)
        train_x, train_y, train_z = to_tensor(train_pool, multi=True)
        test_x, test_y       = to_tensor(test_pool, multi=False)
        return grid_x, grid_y, grid_z, train_x, train_y, train_z, test_x, test_y
