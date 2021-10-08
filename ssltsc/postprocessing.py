import pandas as pd
import os
import pdb
from ssltsc import visualization

def get_mlflow_results(mlflow_id, path='mlruns/'):
    """collect and summarize the mlflow results for one experiment

    Args:
        mlflow_id (int): mlflow id
        path (str, optional): path to the mlruns folder. Defaults to 'mlruns/'.
    """
    path = f'{path}{mlflow_id}/'

    # filter for folders that start with those 32length mlflow hashes
    runs = [run for run in os.listdir(path) if len(run) == 32 and not run.startswith('performance')]

    dict_list = []
    for run in runs:
        # read params
        param_dict = {param: open(f'{path}{run}/params/{param}').read() for param in os.listdir(f'{path}{run}/params/')}
        # read (only test) metrics
        metric_dict = {metric: float(open(f'{path}{run}/metrics/{metric}').read().split(" ")[1]) for metric in os.listdir(f'{path}{run}/metrics/') if metric.startswith('test')}
        # read tags
        tag_dict = {tag: open(f'{path}{run}/tags/{tag}').read() for tag in os.listdir(f'{path}{run}/tags/') if tag not in os.listdir(f'{path}{run}/params/')}
        # combine all dicts in one large dict
        final_dict = {**param_dict, **metric_dict, **tag_dict}
        dict_list.append(final_dict)
    final_frame = pd.DataFrame(dict_list)
    final_frame.to_csv(f'{path}results.csv', index=False)
    print(f'Concatenated and stored results from {len(runs)} runs')

def visualize_experiment(mlflow_id, path='mlruns/'):
    """create boxplots for all runs in one experiment.
    Collects and stores results on the fly if not already done.

    Args:
        mlflow_id (int): mlflow id
        path (str, optional): path to the mlruns folder. Defaults to 'mlruns/'.
    """
    #if not os.path.exists(f'{path}{mlflow_id}/results.csv'):
    get_mlflow_results(mlflow_id=mlflow_id, path=path)
    # visualization.visualize_results_boxplot(mlflow_id=mlflow_id, storage_path=path)
    visualization.visualize_results_lineplot(mlflow_id=mlflow_id, storage_path=path)
