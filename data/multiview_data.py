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
        # Fetch the UCI dataset by ID using the ucimlrepo library
        self.original_data = fetch_ucirepo(id=dataset_id)
        self.X_original = self.original_data.data.features
        self.y_original = self.original_data.data.targets

        # Path to LLM-generated side information (CSV format)
        self.side_info_path = side_info_path

        # Columns used for view1 (original clinical features)
        self.view1_cols = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        self.view1_scaling_cols = ['BMI']

        # Columns used for view2 (LLM-generated privileged info)
        self.view2_cont_cols = [
            'predict_hba1c', 'predict_cholesterol', 'systolic_bp',
            'diastolic_bp', 'exercise_freq', 'hi_sugar_freq'
        ]
        self.view2_cat_cols = ['employment_status']

        self.target = 'Diabetes_binary'
        self.scaler_bmi = StandardScaler()
        self.scaler_view2_cont = StandardScaler()
        self.label_encoders = {col: LabelEncoder() for col in self.view2_cat_cols}

    def preprocess(self):
        # 1) Load the augmented dataset with side information
        augmented_df = pd.read_csv(self.side_info_path)

        # 2) Drop rows with missing values in either features or target
        cols_to_check = self.view2_cont_cols + self.view2_cat_cols + [self.target]
        na_counts = augmented_df[cols_to_check].isna().sum()
        na_ratio = (na_counts / len(augmented_df)) * 100
        col_dtypes = augmented_df[cols_to_check].dtypes
        for col in cols_to_check:
            print(f"{col}: {na_counts[col]} missing ({na_ratio[col]:.2f}%) — dtype: {col_dtypes[col]}")

        cols_to_fill = self.view2_cont_cols + self.view2_cat_cols + [self.target]

        for col in cols_to_fill:
            if augmented_df[col].isna().any():  # NaN이 있는 경우만 처리
                median_value = augmented_df[col].median()
                augmented_df[col].fillna(median_value, inplace=True)

        # augmented_df = augmented_df.dropna(subset=self.view2_cont_cols + self.view2_cat_cols + [self.target])

        # 3) Create a balanced 'grid' set from the training data for hyperparameter tuning
        pos_idx = augmented_df[augmented_df[self.target] == 1].index.tolist()
        neg_idx = augmented_df[augmented_df[self.target] == 0].index.tolist()
        n_pos = min(len(pos_idx), 375)
        n_neg = min(len(neg_idx), 375)
        grid_pos = augmented_df.loc[random.sample(pos_idx, n_pos)]
        grid_neg = augmented_df.loc[random.sample(neg_idx, n_neg)]
        grid_df = pd.concat([grid_pos, grid_neg])

        # Remaining training data used as 'pool'
        train_pool = augmented_df.drop(index=grid_df.index)

        # 4) Construct a separate test set using original dataset (no privileged info)
        original_df = self.X_original.copy()
        original_df[self.target] = self.y_original
        test_pool_df = original_df.drop(index=augmented_df.index, errors='ignore')

        # 5) Fit scaler on BMI and side_info from training pool
        self.scaler_bmi.fit(train_pool[['BMI']])
        def process_view1(df):
            df_copy = df.copy()
            df_copy['BMI'] = self.scaler_bmi.transform(df[['BMI']])
            return df_copy
        grid_df_proc = process_view1(grid_df)
        train_pool_df_proc = process_view1(train_pool)
        test_pool_df_proc = process_view1(test_pool_df)
        # Scale continuous columns in view2 using all augmented training data
        self.scaler_view2_cont.fit(train_pool[self.view2_cont_cols])
        grid_df[self.view2_cont_cols] = self.scaler_view2_cont.transform(grid_df[self.view2_cont_cols])
        train_pool[self.view2_cont_cols] = self.scaler_view2_cont.transform(train_pool[self.view2_cont_cols])

        # 6) Label encode categorical view2 features
        for col in self.view2_cat_cols:
            self.label_encoders[col].fit(augmented_df[col].astype(str))
            grid_df[col] = self.label_encoders[col].transform(grid_df[col].astype(str))
            train_pool[col] = self.label_encoders[col].transform(train_pool[col].astype(str))

        # 7) Combine continuous and categorical view2 features
        grid_view2 = grid_df[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)
        train_view2 = train_pool[self.view2_cont_cols + self.view2_cat_cols].reset_index(drop=True)

        # 8) Convert view2 to PyTorch tensors
        grid_z = torch.tensor(grid_view2.values, dtype=torch.float)
        train_z = torch.tensor(train_view2.values, dtype=torch.float)
        if grid_z.ndim == 1:
            grid_z = grid_z.unsqueeze(1)

        # 9. Convert view1 + label to tensors
        def to_tensor(df, feature_cols, target=None):
            X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            y = torch.tensor(df[target].values, dtype=torch.float32) if target else None
            return X, y

        # Final tensor conversion for all sets
        grid_x, grid_y = to_tensor(grid_df_proc, self.view1_cols, self.target)
        train_x, train_y = to_tensor(train_pool_df_proc, self.view1_cols, self.target)
        test_x, test_y = to_tensor(test_pool_df_proc, self.view1_cols, self.target)

        # 10. Create zero-filled view2 tensor for test set (no privileged info)
        d = grid_z.shape[1] if grid_z.ndim > 1 else 0
        test_z = torch.zeros((test_x.shape[0], d), dtype=torch.float32)

        return (grid_x, grid_y, grid_z), (train_x, train_y, train_z), (test_x, test_y, test_z)
