# data/pairwise_data.py
import json
import pandas as pd
import torch
import numpy as np
from ucimlrepo import fetch_ucirepo
import random

def load_csv(path):
    return pd.read_csv(path)

class PairwisePreprocessor:
    """
    Builds pairwise dataset for diabetes: features, targets, and similarity.
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
            if len(js) < len(group): continue
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
                records.append({**{f'{c}_i': feat_i[idx] for idx,c in enumerate(self.feature_cols)},
                                'target_i': tgt_i,
                                **{f'{c}_j': feat_j[idx] for idx,c in enumerate(self.feature_cols)},
                                'target_j': tgt_j,
                                'similarity': sim})
        df = pd.DataFrame(records)
        before = len(df)
        df = df.dropna()
        after = len(df)
        print(f"Removed {before - after} rows due to NaN values.")
        
        # balanced grid
        pos = df[df['target_i']==1]
        neg = df[df['target_i']==0]
        grid = pd.concat([
            pos.sample(min(len(pos),grid_size_per_class), random_state=42),
            neg.sample(min(len(neg),grid_size_per_class), random_state=42)
        ])
        train_df = df.drop(grid.index)
        # test pool single-instance
        test_orig = self.X_orig.copy()
        test_orig[self.target_col] = self.y_orig
        test_df = test_orig.drop(df['target_i'].index, errors='ignore')
        # tensor conversion
        def to_pair(df_pair):
            Xi = torch.tensor(df_pair[[f'{c}_i' for c in self.feature_cols]].values, dtype=torch.float)
            Xj = torch.tensor(df_pair[[f'{c}_j' for c in self.feature_cols]].values, dtype=torch.float)
            S  = torch.tensor(df_pair['similarity'].values, dtype=torch.float).view(-1,1)
            yi = torch.tensor(df_pair['target_i'].values, dtype=torch.float).view(-1,1)
            yj = torch.tensor(df_pair['target_j'].values, dtype=torch.float).view(-1,1)
            return Xi, Xj, S, yi, yj
        def to_single(df_s):
            X = torch.tensor(df_s[self.feature_cols].values, dtype=torch.float)
            y = torch.tensor(df_s[self.target_col].values, dtype=torch.float).view(-1,1)
            return X, y
        return to_pair(grid), to_pair(train_df), to_single(test_df)