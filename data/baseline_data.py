import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

class BaselineDatasetPreprocessor:
    def __init__(self, dataset_id=891):
        # Fetch the UCI dataset by ID using the ucimlrepo library
        self.original = fetch_ucirepo(id=dataset_id)
        self.X = self.original.data.features  # Feature matrix
        self.y = self.original.data.targets   # Target labels
        self.scaler = StandardScaler()  # Scaler for standardizing features (z-score)

    def preprocess(self):
        # Remove 'ID' column if it exists, to avoid using identifiers as features
        df = self.X.drop(columns=['ID']) if 'ID' in self.X.columns else self.X.copy()

        # Stratified train-test split to preserve label distribution
        train_x, test_x, train_y, test_y = train_test_split(
            df, self.y, test_size=0.25, random_state=42, stratify=self.y
        )

        # Standardize the BMI column in training set
        train_x['BMI'] = self.scaler.fit_transform(train_x[['BMI']])
        train_x['label'] = train_y.values  # Attach labels to training data

        # Balance the dataset by undersampling to 375 samples per class
        pos = train_x[train_x['label'] == 1]
        neg = train_x[train_x['label'] == 0]
        n = min(len(pos), len(neg), 375)
        grid = pd.concat([
            pos.sample(n, random_state=42),
            neg.sample(n, random_state=42)
        ])
        train_pool = train_x.drop(grid.index)  # Remaining training examples not used in 'grid'

        # Standardize the BMI column in test set
        test_x['BMI'] = self.scaler.transform(test_x[['BMI']])
        test_pool = test_x.copy()
        test_pool['label'] = test_y.values # Attach labels to test data

        # Helper function to convert a dataframe to (features, labels) tensors
        def to_tensor(d):
            x = torch.tensor(d.drop(columns=['label']).values, dtype=torch.float)
            y = torch.tensor(d['label'].values, dtype=torch.float)
            return x, y

        # Return 3 sets: grid (balanced training), train_pool (extra training), test
        return (*to_tensor(grid), *to_tensor(train_pool), *to_tensor(test_pool))
