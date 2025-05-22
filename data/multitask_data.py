import random
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

class MultiTaskDatasetPreprocessor:
    """
    Preprocessor for the multi-task model.
    Combines LLM-augmented data with original UCI data.
    Produces train/test splits and auxiliary targets for multi-task learning.
    """
    
    def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
        # Fetch the UCI dataset by ID using the ucimlrepo library
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features  # Original features
        self.y_original = self.original_data.data.targets   # Binary target
        self.scaler = StandardScaler()

        # Path to LLM-generated side information (CSV format)
        self.side_info_path = side_info_path
        
        # Features used for model input
        self.training_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 
            'Income'
        ]
        # Auxiliary and main task labels
        self.target_cols = ['has_diabetes', 'health_1_10', 'diabetes_risk_score', 'Diabetes_binary']
        

    def preprocess(self):
        continuous_cols = ['BMI']
        
        # 1) Load the augmented dataset with side information
        augmented_df = pd.read_csv(self.side_info_path)
        augmented_df = augmented_df[self.training_cols + self.target_cols]
        
        # 2) Drop rows with missing values in either features or target
        # augmented_df = augmented_df.dropna(subset=['has_diabetes', 'health_1_10', 'diabetes_risk_score'])
        augmented_df['has_diabetes'].fillna(0, inplace=True)
        augmented_df['health_1_10'].fillna(augmented_df['health_1_10'].median(), inplace=True)
        augmented_df['diabetes_risk_score'].fillna(augmented_df['diabetes_risk_score'].median(), inplace=True)

        # 3) Create a balanced 'grid' set from the training data for hyperparameter tuning
        pos_idx = augmented_df[augmented_df['Diabetes_binary'] == 1].index.tolist()
        neg_idx = augmented_df[augmented_df['Diabetes_binary'] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
        grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
        grid_df = pd.concat([grid_pos, grid_neg])

        # Remaining training data used as 'pool'
        train_pool = augmented_df.drop(index=grid_df.index)

        # 4) Construct a separate test set using original dataset (no privileged info)
        original_df = self.X_original.copy()
        original_df['Diabetes_binary'] = self.y_original
        test_pool = original_df.drop(index=augmented_df.index, errors='ignore')

        # 5) Fit scaler on BMI and side_info from training pool
        self.scaler.fit(train_pool[continuous_cols])
        grid_df_scaled = grid_df.copy()
        grid_df_scaled[continuous_cols] = self.scaler.transform(grid_df[continuous_cols])
        train_pool_df_scaled = train_pool.copy()
        train_pool_df_scaled[continuous_cols] = self.scaler.transform(train_pool[continuous_cols])
        test_pool_df_scaled = test_pool.copy()
        test_pool_df_scaled[continuous_cols] = self.scaler.transform(test_pool[continuous_cols])

        # 6) Convert to PyTorch tensors
        # --- CONVERT TO TENSORS (multi-task includes auxiliary outputs) ---
        def df_to_tensors_multi(df):
            features = df.drop(columns=self.target_cols, errors='ignore')
            x = torch.from_numpy(features.values).float()
            y_main = torch.from_numpy(df['Diabetes_binary'].values).float()
            y_aux1 = torch.from_numpy(df['health_1_10'].values).float()
            y_aux2 = torch.from_numpy(df['diabetes_risk_score'].values).float()
            y_aux3 = torch.from_numpy(df['has_diabetes'].values).float()
            return x, y_main, y_aux1, y_aux2, y_aux3

        # --- TEST SET (only main task label) ---
        def df_to_tensors_test(df):
            features = df[self.training_cols]
            x = torch.from_numpy(features.values).float()
            y = torch.from_numpy(df['Diabetes_binary'].values).float()
            return x, y

        # Return training + test sets as tensors (grid, pool, test)
        grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3 = df_to_tensors_multi(grid_df_scaled)
        train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool = df_to_tensors_multi(train_pool_df_scaled)
        test_x_pool, test_y_pool = df_to_tensors_test(test_pool_df_scaled)

        return (
            grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3,
            train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
            test_x_pool, test_y_pool
        )
