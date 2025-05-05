# # import random
# # import pandas as pd
# # import torch
# # from sklearn.preprocessing import StandardScaler
# # from ucimlrepo import fetch_ucirepo

# # class MultiViewDatasetPreprocessor:
# #     """
# #     Preprocessor for multi-view learning.
    
# #     - The augmented CSV (side info) is used only for training (grid set and training pool).
# #     - The test data is extracted from the original UCI dataset, and the test set contains only view1 (original features).
# #       (Thus, during testing, instead of view2, a zero tensor with the same dimensions as view2 used in training is created.)
    
# #     Additionally, only the 'BMI' column in view1 is scaled using StandardScaler, and the rest of the columns remain unchanged.
# #     """
# #     def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
# #         self.original_data = fetch_ucirepo(id=dataset_id)
# #         self.X_original = self.original_data.data.features  
# #         self.y_original = self.original_data.data.targets   
# #         self.side_info_path = side_info_path
        
# #         self.view1_cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
# #                            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
# #                            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
# #                            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
# #         self.view1_scaling_cols = ['BMI']
        
# #         # Training view2: side information (only available in the augmented data)
# #         self.view2_cont_cols = ['predict_hba1c', 'predict_cholesterol', 'systolic_bp', 'diastolic_bp', 'exercise_freq', 'hi_sugar_freq']
# #         self.view2_cat_cols = ['employment_status']

# #         self.target_col = 'Diabetes_binary'
        
# #         self.scaler_bmi = StandardScaler() 
# #         self.scaler_view2_cont = StandardScaler()
    
# #     def preprocess(self):
# #         augmented_df = pd.read_csv(self.side_info_path)
# #         augmented_df = augmented_df.dropna(subset=self.view2_cont_cols + self.view2_cat_cols + [self.target_col])
        
# #         # 1. Grid Search Set: Balanced samples (max 375 positive and 375 negative)
# #         pos_idx = augmented_df[augmented_df[self.target_col] == 1].index.tolist()
# #         neg_idx = augmented_df[augmented_df[self.target_col] == 0].index.tolist()
# #         n_pos = min(len(pos_idx), 375)
# #         n_neg = min(len(neg_idx), 375)
# #         grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
# #         grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
# #         grid_df = pd.concat([grid_pos, grid_neg])
        
# #         # 2. Training Pool: Remaining augmented data
# #         train_pool_df = augmented_df.drop(index=grid_df.index)
        
# #         # 3. Test Pool: Remove rows used in the augmented data from the original UCI dataset 
# #         # (augmented data only contains view2 information)
# #         original_df = self.X_original.copy()
# #         original_df[self.target_col] = self.y_original
# #         test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')
        
# #         # 4. Preprocess view1: Scale only 'BMI'
# #         self.scaler_bmi.fit(augmented_df[['BMI']])
# #         def process_view1(df):
# #             df_proc = df.copy()
# #             df_proc['BMI'] = self.scaler_bmi.transform(df[['BMI']])
# #             return df_proc
# #         grid_df_proc = process_view1(grid_df)
# #         train_pool_df_proc = process_view1(train_pool_df)
# #         test_pool_df_proc = process_view1(test_pool_df)
        
# #         # 5. Preprocess view2 continuous: fit on augmented data (grid + training pool)
# #         all_view2_cont = pd.concat([grid_df[self.view2_cont_cols], train_pool_df[self.view2_cont_cols]], axis=0)
# #         self.scaler_view2_cont.fit(all_view2_cont)
# #         grid_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(grid_df[self.view2_cont_cols])
# #         train_pool_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(train_pool_df[self.view2_cont_cols])
        
# #         # 6. One-hot encode view2 categorical columns (augmented data)
# #         grid_df_cat = pd.get_dummies(grid_df[self.view2_cat_cols].astype(str), columns=self.view2_cat_cols)
# #         train_pool_df_cat = pd.get_dummies(train_pool_df[self.view2_cat_cols].astype(str), columns=self.view2_cat_cols)
# #         all_cats = set(grid_df_cat.columns).union(set(train_pool_df_cat.columns))
# #         for c in all_cats:
# #             if c not in grid_df_cat.columns:
# #                 grid_df_cat[c] = 0
# #             if c not in train_pool_df_cat.columns:
# #                 train_pool_df_cat[c] = 0
# #         grid_df_cat = grid_df_cat[list(all_cats)]
# #         train_pool_df_cat = train_pool_df_cat[list(all_cats)]
        
# #         grid_view2 = pd.concat([grid_df[self.view2_cont_cols].reset_index(drop=True),
# #                                 grid_df_cat.reset_index(drop=True)], axis=1)
# #         train_view2 = pd.concat([train_pool_df[self.view2_cont_cols].reset_index(drop=True),
# #                                  train_pool_df_cat.reset_index(drop=True)], axis=1)
        
# #         # 7. Convert data to PyTorch tensors
# #         def df_to_tensor(df, cols, target_col):
# #             X = torch.tensor(df[cols].values, dtype=torch.float32)
# #             y = torch.tensor(df[target_col].values, dtype=torch.float32)
# #             return X, y
        
# #         grid_x, grid_y = df_to_tensor(grid_df_proc, self.view1_cols, self.target_col)
# #         grid_z = torch.tensor(grid_view2.values, dtype=torch.float32)
# #         if grid_z.ndim == 1:
# #             grid_z = grid_z.unsqueeze(1)
        
# #         train_x, train_y = df_to_tensor(train_pool_df_proc, self.view1_cols, self.target_col)
# #         train_z = torch.tensor(train_view2.values, dtype=torch.float32)
# #         if train_z.ndim == 1:
# #             train_z = train_z.unsqueeze(1)
        
# #         test_x, test_y = df_to_tensor(test_pool_df_proc, self.view1_cols, self.target_col)
        
# #         # Since test set does not have view2 information, create a zero tensor with the same dimension as view2.
# #         d = grid_z.shape[1] if grid_z.ndim > 1 else 0
# #         test_z = torch.zeros((test_x.shape[0], d), dtype=torch.float32)
        
# #         return (grid_x, grid_y, grid_z), (train_x, train_y, train_z),(test_x, test_y, test_z)


# import random
# import pandas as pd
# import torch
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from ucimlrepo import fetch_ucirepo

# class MultiViewDatasetPreprocessor:
#     """
#     Preprocessor for multi-view learning without one-hot encoding for view2 categorical features.

#     - The augmented CSV (side info) is used only for training (grid set and training pool).
#     - The test data is extracted from the original UCI dataset, and the test set contains only view1 (original features).
#       A zero tensor is created for view2 in the test set.
#     - Only the 'BMI' column in view1 is scaled; other view1 features remain unchanged.
#     - View2 continuous features are scaled, and categorical features are label-encoded.
#     """
#     def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
#         # Load original UCI data
#         self.original_data = fetch_ucirepo(id=dataset_id)
#         self.X_original = self.original_data.data.features
#         self.y_original = self.original_data.data.targets
#         self.side_info_path = side_info_path

#         # View1 (original features)
#         self.view1_cols = [
#             'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
#             'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#             'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
#             'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
#         ]
#         self.view1_scaling_cols = ['BMI']

#         # View2 (side information)
#         self.view2_cont_cols = [
#             'predict_hba1c', 'predict_cholesterol', 'systolic_bp',
#             'diastolic_bp', 'exercise_freq', 'hi_sugar_freq'
#         ]
#         self.view2_cat_cols = ['employment_status']

#         # Target
#         self.target_col = 'Diabetes_binary'

#         # Scalers and encoders
#         self.scaler_bmi = StandardScaler()
#         self.scaler_view2_cont = StandardScaler()
#         self.label_encoders = {col: LabelEncoder() for col in self.view2_cat_cols}

#     def preprocess(self):
#         # Read augmented data
#         augmented_df = pd.read_csv(self.side_info_path)
#         # Drop rows with missing side-info or targets
#         augmented_df = augmented_df.dropna(subset=self.view2_cont_cols + self.view2_cat_cols + [self.target_col])

#         # 1. Grid Search Set: balanced positive & negative samples (max 375 each)
#         pos_idx = augmented_df[augmented_df[self.target_col] == 1].index.tolist()
#         neg_idx = augmented_df[augmented_df[self.target_col] == 0].index.tolist()
#         n_pos = min(len(pos_idx), 375)
#         n_neg = min(len(neg_idx), 375)
#         grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
#         grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
#         grid_df = pd.concat([grid_pos, grid_neg])

#         # 2. Training Pool: remaining augmented data
#         train_pool_df = augmented_df.drop(index=grid_df.index)

#         # 3. Test Pool: original UCI data excluding augmented indices
#         original_df = self.X_original.copy()
#         original_df[self.target_col] = self.y_original
#         test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')

#         # 4. Preprocess view1: scale only 'BMI'
#         self.scaler_bmi.fit(augmented_df[['BMI']])
#         def process_view1(df):
#             df_copy = df.copy()
#             df_copy['BMI'] = self.scaler_bmi.transform(df[['BMI']])
#             return df_copy
#         grid_df_proc = process_view1(grid_df)
#         train_pool_df_proc = process_view1(train_pool_df)
#         test_pool_df_proc = process_view1(test_pool_df)

#         # 5. Preprocess view2 continuous features
#         all_view2_cont = pd.concat([
#             grid_df[self.view2_cont_cols],
#             train_pool_df[self.view2_cont_cols]
#         ], axis=0)
#         self.scaler_view2_cont.fit(all_view2_cont)
#         grid_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(grid_df[self.view2_cont_cols])
#         train_pool_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(train_pool_df[self.view2_cont_cols])

#         # 6. Label-encode view2 categorical features (no one-hot)
#         for col in self.view2_cat_cols:
#             # Fit on augmented data
#             self.label_encoders[col].fit(augmented_df[col].astype(str))
#             # Transform grid and train
#             grid_df[col] = self.label_encoders[col].transform(grid_df[col].astype(str))
#             train_pool_df[col] = self.label_encoders[col].transform(train_pool_df[col].astype(str))

#         # Combine view2 features
#         grid_view2 = grid_df[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)
#         train_view2 = train_pool_df[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)


#         print("[INFO] View2 Columns (Continuous + Categorical):")
#         print(self.view2_cont_cols + self.view2_cat_cols)
#         print(f"[INFO] Grid View2 DataFrame shape: {grid_view2.shape}")
#         print(f"[INFO] Train View2 DataFrame shape: {train_view2.shape}")

#         # Tensor 변환
#         grid_z = torch.tensor(grid_view2.values, dtype=torch.float32)
#         train_z = torch.tensor(train_view2.values, dtype=torch.float32)

#         print(f"[INFO] Grid View2 Tensor shape (before unsqueeze): {grid_z.shape}")
#         print(grid_z.ndim, grid_z)
#         if grid_z.ndim == 1:
#             print("[WARNING] grid_z was 1D, applying unsqueeze.")
#             grid_z = grid_z.unsqueeze(1)
#         else:
#             print("[INFO] Grid View2 already 2D")

#         print(f"[INFO] Grid View2 Tensor shape (after processing): {grid_z.shape}")

#         # 7. Convert to PyTorch tensors
#         def df_to_tensor(df, feature_cols, target_col=None):
#             X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
#             y = None
#             if target_col:
#                 y = torch.tensor(df[target_col].values, dtype=torch.float32)
#             return X, y

#         grid_x, grid_y = df_to_tensor(grid_df_proc, self.view1_cols, self.target_col)
#         grid_z = torch.tensor(grid_view2.values, dtype=torch.float32)

#         train_x, train_y = df_to_tensor(train_pool_df_proc, self.view1_cols, self.target_col)
#         train_z = torch.tensor(train_view2.values, dtype=torch.float32)

#         test_x, test_y = df_to_tensor(test_pool_df_proc, self.view1_cols, self.target_col)
#         # Create zero tensor for view2 in test set
#         d = grid_z.shape[1] if grid_z.ndim > 1 else 0
#         test_z = torch.zeros((test_x.shape[0], d), dtype=torch.float32)

#         return (grid_x, grid_y, grid_z), (train_x, train_y, train_z), (test_x, test_y, test_z)

# import random
# import pandas as pd
# import torch
# from sklearn.preprocessing import StandardScaler
# from ucimlrepo import fetch_ucirepo

# class MultiViewDatasetPreprocessor:
#     """
#     Preprocessor for multi-view learning.
    
#     - The augmented CSV (side info) is used only for training (grid set and training pool).
#     - The test data is extracted from the original UCI dataset, and the test set contains only view1 (original features).
#       (Thus, during testing, instead of view2, a zero tensor with the same dimensions as view2 used in training is created.)
    
#     Additionally, only the 'BMI' column in view1 is scaled using StandardScaler, and the rest of the columns remain unchanged.
#     """
#     def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
#         self.original_data = fetch_ucirepo(id=dataset_id)
#         self.X_original = self.original_data.data.features  
#         self.y_original = self.original_data.data.targets   
#         self.side_info_path = side_info_path
        
#         self.view1_cols = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
#                            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#                            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
#                            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
#         self.view1_scaling_cols = ['BMI']
        
#         # Training view2: side information (only available in the augmented data)
#         self.view2_cont_cols = ['predict_hba1c', 'predict_cholesterol', 'systolic_bp', 'diastolic_bp', 'exercise_freq', 'hi_sugar_freq']
#         self.view2_cat_cols = ['employment_status']

#         self.target_col = 'Diabetes_binary'
        
#         self.scaler_bmi = StandardScaler() 
#         self.scaler_view2_cont = StandardScaler()
    
#     def preprocess(self):
#         augmented_df = pd.read_csv(self.side_info_path)
#         augmented_df = augmented_df.dropna(subset=self.view2_cont_cols + self.view2_cat_cols + [self.target_col])
        
#         # 1. Grid Search Set: Balanced samples (max 375 positive and 375 negative)
#         pos_idx = augmented_df[augmented_df[self.target_col] == 1].index.tolist()
#         neg_idx = augmented_df[augmented_df[self.target_col] == 0].index.tolist()
#         n_pos = min(len(pos_idx), 375)
#         n_neg = min(len(neg_idx), 375)
#         grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
#         grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
#         grid_df = pd.concat([grid_pos, grid_neg])
        
#         # 2. Training Pool: Remaining augmented data
#         train_pool_df = augmented_df.drop(index=grid_df.index)
        
#         # 3. Test Pool: Remove rows used in the augmented data from the original UCI dataset 
#         # (augmented data only contains view2 information)
#         original_df = self.X_original.copy()
#         original_df[self.target_col] = self.y_original
#         test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')
        
#         # 4. Preprocess view1: Scale only 'BMI'
#         self.scaler_bmi.fit(augmented_df[['BMI']])
#         def process_view1(df):
#             df_proc = df.copy()
#             df_proc['BMI'] = self.scaler_bmi.transform(df[['BMI']])
#             return df_proc
#         grid_df_proc = process_view1(grid_df)
#         train_pool_df_proc = process_view1(train_pool_df)
#         test_pool_df_proc = process_view1(test_pool_df)
        
#         # 5. Preprocess view2 continuous: fit on augmented data (grid + training pool)
#         all_view2_cont = pd.concat([grid_df[self.view2_cont_cols], train_pool_df[self.view2_cont_cols]], axis=0)
#         self.scaler_view2_cont.fit(all_view2_cont)
#         grid_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(grid_df[self.view2_cont_cols])
#         train_pool_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(train_pool_df[self.view2_cont_cols])
        
#         # 6. One-hot encode view2 categorical columns (augmented data)
#         grid_df_cat = pd.get_dummies(grid_df[self.view2_cat_cols].astype(str), columns=self.view2_cat_cols)
#         train_pool_df_cat = pd.get_dummies(train_pool_df[self.view2_cat_cols].astype(str), columns=self.view2_cat_cols)
#         all_cats = set(grid_df_cat.columns).union(set(train_pool_df_cat.columns))
#         for c in all_cats:
#             if c not in grid_df_cat.columns:
#                 grid_df_cat[c] = 0
#             if c not in train_pool_df_cat.columns:
#                 train_pool_df_cat[c] = 0
#         grid_df_cat = grid_df_cat[list(all_cats)]
#         train_pool_df_cat = train_pool_df_cat[list(all_cats)]
        
#         grid_view2 = pd.concat([grid_df[self.view2_cont_cols].reset_index(drop=True),
#                                 grid_df_cat.reset_index(drop=True)], axis=1)
#         train_view2 = pd.concat([train_pool_df[self.view2_cont_cols].reset_index(drop=True),
#                                  train_pool_df_cat.reset_index(drop=True)], axis=1)
        
#         # 7. Convert data to PyTorch tensors
#         def df_to_tensor(df, cols, target_col):
#             X = torch.tensor(df[cols].values, dtype=torch.float32)
#             y = torch.tensor(df[target_col].values, dtype=torch.float32)
#             return X, y
        
#         grid_x, grid_y = df_to_tensor(grid_df_proc, self.view1_cols, self.target_col)
#         grid_z = torch.tensor(grid_view2.values, dtype=torch.float32)
#         if grid_z.ndim == 1:
#             grid_z = grid_z.unsqueeze(1)
        
#         train_x, train_y = df_to_tensor(train_pool_df_proc, self.view1_cols, self.target_col)
#         train_z = torch.tensor(train_view2.values, dtype=torch.float32)
#         if train_z.ndim == 1:
#             train_z = train_z.unsqueeze(1)
        
#         test_x, test_y = df_to_tensor(test_pool_df_proc, self.view1_cols, self.target_col)
        
#         # Since test set does not have view2 information, create a zero tensor with the same dimension as view2.
#         d = grid_z.shape[1] if grid_z.ndim > 1 else 0
#         test_z = torch.zeros((test_x.shape[0], d), dtype=torch.float32)
        
#         return (grid_x, grid_y, grid_z), (train_x, train_y, train_z),(test_x, test_y, test_z)


import random
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo

class MultiViewDatasetPreprocessor:
    """
    Preprocessor for multi-view learning without one-hot encoding for view2 categorical features.

    - The augmented CSV (side info) is used only for training (grid set and training pool).
    - The test data is extracted from the original UCI dataset, and the test set contains only view1 (original features).
      A zero tensor is created for view2 in the test set.
    - Only the 'BMI' column in view1 is scaled; other view1 features remain unchanged.
    - View2 continuous features are scaled, and categorical features are label-encoded.
    """
    def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
        # Load original UCI data
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features
        self.y_original = self.original_data.data.targets
        self.side_info_path = side_info_path

        # View1 (original features)
        self.view1_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        self.view1_scaling_cols = ['BMI']

        # View2 (side information)
        self.view2_cont_cols = [
            'predict_hba1c', 'predict_cholesterol', 'systolic_bp',
            'diastolic_bp', 'exercise_freq', 'hi_sugar_freq'
        ]
        self.view2_cat_cols = ['employment_status']

        # Target
        self.target_col = 'Diabetes_binary'

        # Scalers and encoders
        self.scaler_bmi = StandardScaler()
        self.scaler_view2_cont = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in self.view2_cat_cols}

    def preprocess(self):
        # Read augmented data
        augmented_df = pd.read_csv(self.side_info_path)
        # Drop rows with missing side-info or targets
        augmented_df = augmented_df.dropna(subset=self.view2_cont_cols + self.view2_cat_cols + [self.target_col])

        # 1. Grid Search Set: balanced positive & negative samples (max 375 each)
        pos_idx = augmented_df[augmented_df[self.target_col] == 1].index.tolist()
        neg_idx = augmented_df[augmented_df[self.target_col] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
        grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
        grid_df = pd.concat([grid_pos, grid_neg])

        # 2. Training Pool: remaining augmented data
        train_pool_df = augmented_df.drop(index=grid_df.index)

        # 3. Test Pool: original UCI data excluding augmented indices
        original_df = self.X_original.copy()
        original_df[self.target_col] = self.y_original
        test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')

        # 4. Preprocess view1: scale only 'BMI'
        self.scaler_bmi.fit(augmented_df[['BMI']])
        def process_view1(df):
            df_copy = df.copy()
            df_copy['BMI'] = self.scaler_bmi.transform(df[['BMI']])
            return df_copy
        grid_df_proc = process_view1(grid_df)
        train_pool_df_proc = process_view1(train_pool_df)
        test_pool_df_proc = process_view1(test_pool_df)

        # 5. Preprocess view2 continuous features
        all_view2_cont = pd.concat([
            grid_df[self.view2_cont_cols],
            train_pool_df[self.view2_cont_cols]
        ], axis=0)
        self.scaler_view2_cont.fit(all_view2_cont)
        grid_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(grid_df[self.view2_cont_cols])
        train_pool_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(train_pool_df[self.view2_cont_cols])

        # 6. Label-encode view2 categorical features (no one-hot)
        for col in self.view2_cat_cols:
            # Fit on augmented data
            self.label_encoders[col].fit(augmented_df[col].astype(str))
            # Transform grid and train
            grid_df[col] = self.label_encoders[col].transform(grid_df[col].astype(str))
            train_pool_df[col] = self.label_encoders[col].transform(train_pool_df[col].astype(str))

        # Combine view2 features
        grid_view2 = grid_df[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)
        train_view2 = train_pool_df[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)


        print("[INFO] View2 Columns (Continuous + Categorical):")
        print(self.view2_cont_cols + self.view2_cat_cols)
        print(f"[INFO] Grid View2 DataFrame shape: {grid_view2.shape}")
        print(f"[INFO] Train View2 DataFrame shape: {train_view2.shape}")

        # Tensor 변환
        grid_z = torch.tensor(grid_view2.values, dtype=torch.float32)
        train_z = torch.tensor(train_view2.values, dtype=torch.float32)

        print(f"[INFO] Grid View2 Tensor shape (before unsqueeze): {grid_z.shape}")
        print(grid_z.ndim, grid_z)
        if grid_z.ndim == 1:
            print("[WARNING] grid_z was 1D, applying unsqueeze.")
            grid_z = grid_z.unsqueeze(1)
        else:
            print("[INFO] Grid View2 already 2D")

        print(f"[INFO] Grid View2 Tensor shape (after processing): {grid_z.shape}")

        # 7. Convert to PyTorch tensors
        def df_to_tensor(df, feature_cols, target_col=None):
            X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            y = None
            if target_col:
                y = torch.tensor(df[target_col].values, dtype=torch.float32)
            return X, y

        grid_x, grid_y = df_to_tensor(grid_df_proc, self.view1_cols, self.target_col)
        grid_z = torch.tensor(grid_view2.values, dtype=torch.float32)

        train_x, train_y = df_to_tensor(train_pool_df_proc, self.view1_cols, self.target_col)
        train_z = torch.tensor(train_view2.values, dtype=torch.float32)

        test_x, test_y = df_to_tensor(test_pool_df_proc, self.view1_cols, self.target_col)
        # Create zero tensor for view2 in test set
        d = grid_z.shape[1] if grid_z.ndim > 1 else 0
        test_z = torch.zeros((test_x.shape[0], d), dtype=torch.float32)

        return (grid_x, grid_y, grid_z), (train_x, train_y, train_z), (test_x, test_y, test_z)
