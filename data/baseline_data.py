import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

class BaselineDatasetPreprocessor:
    def __init__(self, dataset_id=891):
        repo = fetch_ucirepo(id=dataset_id)
        self.X = repo.data.features
        self.y = repo.data.targets
        self.scaler = StandardScaler()

    def preprocess(self):
        df = self.X.drop(columns=['ID']) if 'ID' in self.X.columns else self.X.copy()
        tr, te, y_tr, y_te = train_test_split(
            df, self.y, test_size=0.25, random_state=42, stratify=self.y
        )
        tr['BMI'] = self.scaler.fit_transform(tr[['BMI']])
        tr['label'] = y_tr.values

        pos = tr[tr['label']==1]
        neg = tr[tr['label']==0]
        n = min(len(pos), len(neg), 375)
        grid = pd.concat([pos.sample(n, random_state=42), neg.sample(n, random_state=42)])
        pool = tr.drop(grid.index)

        te['BMI'] = self.scaler.transform(te[['BMI']])
        te['label'] = y_te.values

        def to_xy(d):
            x = torch.tensor(d.drop(columns=['label']).values, dtype=torch.float)
            y = torch.tensor(d['label'].values, dtype=torch.float)
            return x, y

        return (*to_xy(grid), *to_xy(pool), *to_xy(te))