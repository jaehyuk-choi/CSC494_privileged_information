import json
import itertools
import subprocess

# List of experiment configs (same as in your main script)
experiments = [
    # {
    #     "name": "MT-MLP",
    #     "model": "MultiTaskNN",
    #     "param_grid": {
    #         "lr": [0.01, 0.1],
    #         "hidden_dim": [64, 128, 256],
    #         "num_layers": [1, 2, 3, 4],
    #         "lambda_aux": [0.01, 0.1, 0.3]
    #     }
    # },
    # {
    #     "name": "PretrainFT",
    #     "model": "MultiTaskNN_PretrainFinetuneExtended",
    #     "param_grid": {
    #         "lr_pre": [0.01, 0.1],
    #         "lr_fine": [0.01, 0.1],
    #         "num_layers": [1, 2, 3, 4],
    #         "hidden_dim": [64, 128, 256],
    #         "lambda_aux": [0.01, 0.1, 0.3],
    #         "pre_epochs": [100, 300],
    #         "fine_epochs": [100, 300]
    #     }
    # },
    # {
    #     "name": "Decoupled",
    #     "model": "MultiTaskNN_Decoupled",
    #     "param_grid": {
    #         "lr": [0.01, 0.1],
    #         "hidden_dim": [64, 128, 256],
    #         "num_layers": [1, 2, 3, 4],
    #         "lambda_aux": [0.01, 0.1, 0.3]
    #     }
    # }
    {
        "name": "MT-MLP",
        "model": "MultiTaskNN",
        "param_grid": {
            "lr": [0.1],
            "hidden_dim": [64],
            "num_layers": [1,2,3,4],
            "lambda_aux": [0.01, 0.1]
        }
    },
    {
        "name": "PretrainFT",
        "model": "MultiTaskNN_PretrainFinetuneExtended",
        "param_grid": {
            "lr_pre": [0.1],
            "lr_fine": [0.1],
            "num_layers": [1, 2, 3, 4],
            "hidden_dim": [64],
            "lambda_aux": [0.01, 0.1],
            "pre_epochs": [100],
            "fine_epochs": [100]
        }
    },
    {
        "name": "Decoupled",
        "model": "MultiTaskNN_Decoupled",
        "param_grid": {
            "lr": [0.01],
            "hidden_dim": [64],
            "num_layers": [1, 2, 3, 4],
            "lambda_aux": [0.01, 0.1]
        }
    }
]

SBATCH_SCRIPT = 'run_multitask.sh'
NUM_RUNS = 10
BASE_SEED = 42


for exp in experiments:
    name       = exp["name"]
    model      = exp["model"]
    param_grid = exp["param_grid"]

    keys   = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        payload = {
            "name":  name,
            "model": model,
            "params": params
        }
        payload_json = json.dumps(payload)

        job_name = name + "_" + "_".join(f"{k}{v}" for k,v in params.items())

        print(f"Submitting job {job_name}")
        subprocess.run([
            "sbatch",
            "--job-name", job_name,
            SBATCH_SCRIPT,
            payload_json
        ], check=True)