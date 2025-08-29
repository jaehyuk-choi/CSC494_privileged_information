import json
import itertools
import subprocess
import argparse

# Usage:
#   python grid_submit.py --config params.json --script run_model.sh

def load_configs(path):
    with open(path, 'r') as fp:
        return json.load(fp)

def build_combinations(param_grid):
    """
    Given a dict of parameter lists, return list of dicts for each combination.
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos

def submit_jobs(configs, script):
    """
    For each model config, create sbatch jobs for all parameter combinations.
    """
    for cfg in configs:
        name = cfg['name']
        grid = cfg['param_grid']
        combos = build_combinations(grid)
        for params in combos:
            # build unique job name
            parts = [f"{k}{v}" for k, v in params.items()]
            jobname = f"{name}_" + "_".join(parts)
            # build argument list: model name + all param values in grid order
            arg_vals = [str(params[k]) for k in grid.keys()]
            cmd = ['sbatch', '--job-name', jobname, script, name] + arg_vals
            print("Submitting:", ' '.join(cmd))
            subprocess.run(cmd, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grid search sbatch submitter")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config with model grid definitions')
    parser.add_argument('--script', type=str, default='run_model.sh',
                        help='SBATCH wrapper script to call for each job')
    args = parser.parse_args()

    configs = load_configs(args.config)
    submit_jobs(configs, args.script)