# scripts/pairwise.py
import torch
from data.pairwise_data import PairwisePreprocessor
from model.pairwise_model.soft import PairwiseDiabetesModel_Soft
from model.pairwise_model.continuous import PairwiseDiabetesModel_Continuous
from sklearn.model_selection import KFold
from utils import set_seed, grid_search_cv_pairwise, run_experiments_pairwise
import numpy as np
import csv

def main():
    set_seed(42)
    feat_cols=[ 'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
        'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
        'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
        'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']
    prep=PairwisePreprocessor('prompting/augmented_data_pairwise_8B.csv',
                              'prompting/pairs.json',891,feat_cols,'Diabetes_binary')
    grid_set, train_set, test_set=prep.preprocess()

    grid={'margin':[1.0],'lambda_w':[0.3,0.5,0.7],'gamma':[1.0],
          'hidden_dim':[32,64],'lr':[1e-3],'epochs':[100]}

    for name,model in [('Soft',PairwiseDiabetesModel_Soft),
                       ('Cont',PairwiseDiabetesModel_Continuous)]:
        best,auc=grid_search_cv_pairwise(model,grid,grid_set)
        print(f"{name} best: {best}, AUC={auc:.3f}")
        results=run_experiments_pairwise(model,best,train_set,test_set)
        agg={k:f"{np.mean([r[k] for r in results]):.2f}±{np.std([r[k] for r in results]):.2f}" for k in results[0]}
        with open(f"results_pairwise_{name}.csv",'w',newline='') as f:
            w=csv.writer(f); w.writerow(['Metric','Value']); w.writerows(agg.items())

if __name__=='__main__': main()