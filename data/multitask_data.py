import random
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# class MultiTaskDatasetPreprocessor:
#     """
#     Preprocessor for multi-task experiments.
#     - Loads original UCI data and side-information CSV.
#     - Splits into grid-search set (375 pos + 375 neg), training pool, and test pool.
#     - Scales continuous features and converts to torch.Tensors.
#     """
#     def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
#         self.dataset_id = dataset_id
#         self.side_info_path = side_info_path
#         # fetch original features and labels
#         self.original_data = fetch_ucirepo(id=self.dataset_id)
#         self.X_original = self.original_data.data.features  # pandas DataFrame
#         self.y_original = self.original_data.data.targets   # pandas Series
#         self.scaler = StandardScaler()

#     def preprocess(self):
#         # 1. Load and clean side-information
#         continuous_cols = ['BMI']
#         augmented_df = pd.read_csv(self.side_info_path)

#         # Define training and target columns
#         training_cols = [
#             'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
#             'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#             'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
#             'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
#             'Income'
#         ]
#         target_cols = ['has_diabetes', 'health_1_10', 'diabetes_risk_score', 'Diabetes_binary']

#         # Keep only relevant columns
#         augmented_df = augmented_df[training_cols + target_cols]

#         # Fill missing values
#         augmented_df['has_diabetes'].fillna(0, inplace=True)
#         augmented_df['health_1_10'].fillna(augmented_df['health_1_10'].median(), inplace=True)
#         augmented_df['diabetes_risk_score'].fillna(augmented_df['diabetes_risk_score'].median(), inplace=True)

#         # 2. Create grid-search set: 375 positive + 375 negative examples
#         pos_idx = augmented_df[augmented_df['Diabetes_binary'] == 1].index.tolist()
#         print(pos_idx)
#         neg_idx = augmented_df[augmented_df['Diabetes_binary'] == 0].index.tolist()
#         n_pos = min(len(pos_idx), 375)
#         n_neg = min(len(neg_idx), 375)
#         grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
#         grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
#         grid_df = pd.concat([grid_pos, grid_neg])

#         # 3. Remaining data → training pool
#         train_pool_df = augmented_df.drop(index=grid_df.index)
#         # 4. Original data minus any augmented indices → test pool
#         original_df = self.X_original.copy()
#         original_df['Diabetes_binary'] = self.y_original
#         test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')

#         # 5. Scale continuous columns
#         self.scaler.fit(augmented_df[continuous_cols])
#         grid_df_scaled = grid_df.copy()
#         grid_df_scaled[continuous_cols] = self.scaler.transform(grid_df[continuous_cols])
        
#         train_pool_df_scaled = train_pool_df.copy()
#         train_pool_df_scaled[continuous_cols] = self.scaler.transform(train_pool_df[continuous_cols])

#         test_pool_df_scaled = test_pool_df.copy()
#         test_pool_df_scaled[continuous_cols] = self.scaler.transform(test_pool_df[continuous_cols])
        
#         def df_to_tensors_multi(df):
#             feats = df.drop(columns=target_cols, errors='ignore')
#             x = torch.from_numpy(feats.values).float()
#             y_main = torch.from_numpy(df['Diabetes_binary'].values).float()
#             y_aux1 = torch.from_numpy(df['health_1_10'].values).float()
#             y_aux2 = torch.from_numpy(df['diabetes_risk_score'].values).float()
#             y_aux3 = torch.from_numpy(df['has_diabetes'].values).float()
#             return x, y_main, y_aux1, y_aux2, y_aux3

#         # Helper: convert test-only df to tensors
#         def df_to_tensors_test(df):
#             training_cols = [
#             'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
#             'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
#             'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
#             'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 
#             'Income'
#             ]
#             features = df[training_cols]
#             x = torch.from_numpy(features.values).float()
#             y = torch.from_numpy(df['Diabetes_binary'].values).float()
#             return x, y

#         # Produce final tensors
#         grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3 = df_to_tensors_multi(grid_df)
#         train_x, train_y, train_aux1, train_aux2, train_aux3 = df_to_tensors_multi(train_pool_df)
#         test_x, test_y = df_to_tensors_test(test_pool_df)
#         return (
#             grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3,
#             train_x, train_y, train_aux1, train_aux2, train_aux3,
#             test_x, test_y
#         )

class MultiTaskDatasetPreprocessor:
    """
    Preprocessor for the multi-task model.
    Uses an augmented CSV file along with the original UCI data.
    Data is scaled, balanced, and converted to PyTorch tensors.
    """
    def __init__(self, dataset_id=891, side_info_path='prompting/augmented_data.csv'):
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features  # pandas DataFrame
        self.y_original = self.original_data.data.targets   # pandas Series
        self.side_info_path = side_info_path
        self.scaler = StandardScaler()
        
    def preprocess(self):
        continuous_cols = ['BMI']
        augmented_df = pd.read_csv(self.side_info_path)
        
        training_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 
            'Income'
        ]
        # training_cols = [
        #     'CholCheck', 'Smoker', 'Stroke',
        #     'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        #     'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
        #     'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Education', 'Income'
        # ]
        target_cols = ['has_diabetes', 'health_1_10', 'diabetes_risk_score', 'Diabetes_binary']
        augmented_df = augmented_df[training_cols + target_cols]
        
        augmented_df['has_diabetes'].fillna(0, inplace=True)
        augmented_df['health_1_10'].fillna(augmented_df['health_1_10'].median(), inplace=True)
        augmented_df['diabetes_risk_score'].fillna(augmented_df['diabetes_risk_score'].median(), inplace=True)
        
        # Grid Search set: balanced sampling
        # grid_size = 750
        # grid_df = augmented_df.sample(n=grid_size, random_state=SEED)
        pos_idx = augmented_df[augmented_df['Diabetes_binary'] == 1].index.tolist()
        neg_idx = augmented_df[augmented_df['Diabetes_binary'] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
        grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
        grid_df = pd.concat([grid_pos, grid_neg])
        
        # Training Pool: remaining data
        train_pool_df = augmented_df.drop(index=grid_df.index)
        
        # Test Pool: remove augmented_df indices from original data
        original_df = self.X_original.copy()
        original_df['Diabetes_binary'] = self.y_original
        test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')
        
        self.scaler.fit(augmented_df[continuous_cols])
        grid_df_scaled = grid_df.copy()
        grid_df_scaled[continuous_cols] = self.scaler.transform(grid_df[continuous_cols])
    
        train_pool_df_scaled = train_pool_df.copy()
        train_pool_df_scaled[continuous_cols] = self.scaler.transform(train_pool_df[continuous_cols])
        
        test_pool_df_scaled = test_pool_df.copy()
        test_pool_df_scaled[continuous_cols] = self.scaler.transform(test_pool_df[continuous_cols])
        
        def df_to_tensors_multi(df):
            features = df.drop(columns=target_cols, errors='ignore')
            x = torch.from_numpy(features.values).float()
            y_main = torch.from_numpy(df['Diabetes_binary'].values).float()
            y_aux1 = torch.from_numpy(df['health_1_10'].values).float()
            y_aux2 = torch.from_numpy(df['diabetes_risk_score'].values).float()
            y_aux3 = torch.from_numpy(df['has_diabetes'].values).float()
            return x, y_main, y_aux1, y_aux2, y_aux3
        
        def df_to_tensors_test(df):
            training_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 
            'Income'
        ]
            features = df[training_cols]
            x = torch.from_numpy(features.values).float()
            y = torch.from_numpy(df['Diabetes_binary'].values).float()
            return x, y
        
        grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3 = df_to_tensors_multi(grid_df_scaled)
        train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool = df_to_tensors_multi(train_pool_df_scaled)
        test_x_pool, test_y_pool = df_to_tensors_test(test_pool_df_scaled)
        print(train_pool_df_scaled.index)
        return (grid_x, grid_y, grid_aux1, grid_aux2, grid_aux3,
                train_x_pool, train_y_pool, train_aux1_pool, train_aux2_pool, train_aux3_pool,
                test_x_pool, test_y_pool)
