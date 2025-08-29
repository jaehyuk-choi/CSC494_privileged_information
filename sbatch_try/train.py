import argparse
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def main(args):
    # Load large dataset
    data = fetch_covtype()
    X, y = data.data, data.target

    # Define model
    model = GradientBoostingClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    # Cross-validation with shuffle to create different splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"CV mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GradientBoosting on CoverType dataset")
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--n_estimators', type=int, required=True)
    parser.add_argument('--max_depth', type=int, required=True)
    args = parser.parse_args()
    main(args)
