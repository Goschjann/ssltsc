import argparse
import os
import tempfile
import yaml
import pdb
import optuna
import mlflow
from pandas import DataFrame
from optuna import Trial
import plotly
plotly.io.kaleido.scope.default_format

def convert_to_best_config(config: dict, best_params: dict):
    best_config = config.copy()
    for section, params in best_config.items():
        if section == 'search_space':
            continue

        for hp_name in best_config[section].keys():
            if hp_name in best_params.keys():
                config[section][hp_name] = best_params[hp_name]

    return best_config

def convert_to_tuner_config(config: dict, trial: Trial):
    search_space = config['search_space']
    config = config.copy()

    # Find and replace the hyper parameters defined in search_space
    for section, params in config.items():
        if section == 'search_space':
            continue

        for hp_name in config[section].keys():
            if hp_name in search_space.keys():
                if hp_name == 'loss_weights':
                    low = search_space[hp_name]['low']
                    high = search_space[hp_name]['high']
                    suggested_losses = [
                        trial.suggest_loguniform("lw_{}".format(lw), low=low,
                                                 high=high) for lw in
                        range(search_space[hp_name]['length'])]
                    config[section][hp_name] = suggested_losses
                    continue

                tuner_hp_type = search_space[hp_name]['type']
                if tuner_hp_type == 'categorical':
                    config[section][hp_name] = trial.suggest_categorical(hp_name,
                                        choices=search_space[hp_name]['choices'])
                    continue

                low = search_space[hp_name]['low']
                high = search_space[hp_name]['high']
                if tuner_hp_type == 'float':
                    config[section][hp_name] = trial.suggest_float(hp_name, low=low, high=high,
                                                                   step=search_space[hp_name]['step'])
                elif tuner_hp_type == 'int':
                    config[section][hp_name] = trial.suggest_int(hp_name, low=low, high=high,
                                                                 step=search_space[hp_name]['step'])
                elif tuner_hp_type == 'log':
                    config[section][hp_name] = trial.suggest_loguniform(hp_name, low=low, high=high)
                else:
                    raise ValueError("Expected the tuner hyper parameter to have type int or float")
    return config

def log_optuna_plots_in_mlflow(study):
    """store optuna plots and log them in mlflow
    """
    with tempfile.TemporaryDirectory() as tmp_dir:

        plot = optuna.visualization.plot_slice(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/sliceplot.png')
        mlflow.log_artifact(f'{tmp_dir}/sliceplot.png')

        plot = optuna.visualization.plot_intermediate_values(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/interm_values.png')
        mlflow.log_artifact(f'{tmp_dir}/interm_values.png')

        plot = optuna.visualization.plot_optimization_history(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/history.png')
        mlflow.log_artifact(f'{tmp_dir}/history.png')

        plot = optuna.visualization.plot_contour(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/contour.png')
        mlflow.log_artifact(f'{tmp_dir}/contour.png')

        plot = optuna.visualization.plot_param_importances(study=study)
        plotly.io.write_image(fig=plot, file=f'{tmp_dir}/importances.png')
        mlflow.log_artifact(f'{tmp_dir}/importances.png')


def save_as_csv_file_in_mlflow(data: DataFrame, filename: str):
    """save csv file
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        data.to_csv(f'{tmp_dir}/{filename}')
        mlflow.log_artifact(f'{tmp_dir}/{filename}')


def save_as_yaml_file_in_mlflow(data: dict, filename: str):

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)
        with open(path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        mlflow.log_artifact(path)


def get_or_create_experiment_id(experiment_name: str) -> int:
    """
    Lookup experiment name in mlflow database. If the experiment name
    does not exist then it will create a new experiment.

    Args:
        experiment_name:

    Returns:
        mlflow experiment id
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    elif experiment.lifecycle_stage == 'deleted':
        print("The experiment name exists in the .trash. Delete .trash or come up with other name.")

    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def get_experiment_id(config=None, args=None):
    """Given parsed arguments this method will look for
    the `mlflow_name` and `mlflow_id` arguments.
    Either a config file OR an args object should be passed.
    We use configs in the modeling files but need the args functionality for the postprocessing

    Args:
        config: A dict of configurations
        args: Command line args as read by arg parse

    Returns:
        A mlflow experiment id
    """
    if args is not None:
        if not 'mlflow_name' in args or not 'mlflow_id' in args:
            raise KeyError("The args need to have a mlflow_name and a mlflow_id")

        if args.mlflow_name:
            experiment_id = get_or_create_experiment_id(args.mlflow_name)

        elif args.mlflow_id:
            experiment_id = args.mlflow_id

        else:
            # Neither a mlflow name or id is given. We create a default name.
            default_experiment_name = "default_experiment"
            experiment_id = get_or_create_experiment_id(default_experiment_name)

        return experiment_id
    elif config is not None:
        params = config['exp_params']
        if not 'mlflow_name' in params or not 'mlflow_id' in params:
            raise KeyError("The args need to have a mlflow_name and a mlflow_id")

        if params['mlflow_name']:
            experiment_id = get_or_create_experiment_id(params['mlflow_name'])

        elif params['mlflow_id'] and params['mlflow_id'] > 1:
            experiment_id = params['mlflow_id']

        else:
            # Neither a mlflow name or id is given. We create a default name.
            default_experiment_name = "default_experiment"
            experiment_id = get_or_create_experiment_id(default_experiment_name)

    return experiment_id


def save_history_to_mlflow(history: DataFrame):
    """
    Logs all metrics in a history DataFrame per step in mlflow
    and saves the DataFrame as a .csv file artifact in mlflow.
    Args:
        history:
            pandas.DataFrame
    """
    assert 'step' in history, "To save a history in mlflow a `step` column is needed"

    for index, row in history.iterrows():
        mlflow.log_metrics(row.drop('step').to_dict(), step=int(row['step']))

    with tempfile.TemporaryDirectory() as tmp_dir:
        history_path = os.path.join(tmp_dir, "history.csv")
        history.to_csv(history_path)
        mlflow.log_artifact(history_path)



def get_base_argparser(description=None):
    parser = argparse.ArgumentParser(description=description)

    # General configurations
    parser.add_argument('--mlflow_name', type=str, default=None,
                        help='mlflow experiment name, overwrites mlflow_id')
    parser.add_argument('--mlflow_id', type=int, default=None, metavar=None,
                        help='mlflow experiment id')
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='seed (default: 1337)')
    parser.add_argument('--dataset', type=str, default=None, metavar='N',
                        help='used tsc dataset (default: pamap2)')
    parser.add_argument('--path', type=str, default=None,
                        help='path of the stored data')
    parser.add_argument('--verbose', action='store_true',
                        help='print details when running script')
    parser.add_argument('--early_stopping', action='store_true', default=None,
                        help='ES Y/N')
    parser.add_argument('--early_stopping_metric', type=str, default=None,
                        help='metric to control early stopping, only takes effect if early_stopping flag')
    parser.add_argument('--sample_supervised', action='store_true', default=None,
                         help='Use a supervised (yes) or a semi-supervised (no) training data sampler')
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='path to config file',
                        default='config_files/supervised.yaml')

    # Training details
    parser.add_argument('--batch_size', type=int, default=None, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--n_steps', type=int, default=None, metavar='N',
                        help='amount of training steps (default: 1000)')
    parser.add_argument('--lr_scheduler', type=str, default=None, metavar='N',
                        help='lr scheduler to use. (default: None')
    parser.add_argument('--inference_batch_size', type=int, default=None, metavar='N',
                        help='batch size when doing prediction (default: 64)')
    parser.add_argument('--val_steps', type=int, default=None, metavar='N',
                        help='frequency of validation steps')

    # Model details
    parser.add_argument('--backbone', type=str, default=None,
                        help='backbone architecture, for cifar10: convnet13, wideresnet18, for tsc: fcn')
    parser.add_argument('--model_name', type=str, default=None,
                        help='custom model name. this will be set to the the default model name if None (default: None)')
    parser.add_argument('--lr', type=float, default=None, metavar='N',
                        help='learning rate for optimizer (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=None, metavar='N',
                        help='weight decay for optimizer (default: 0.0)')

    # Semi-supervised settings
    parser.add_argument('--num_labels', type=int, default=None, metavar='N',
                        help='amount of labeled samples (default: 250)')
    parser.add_argument('--labeled_batch_size', type=int, default=None, metavar='N',
                        help='number of labeled samples per batch. (default: 16)')

    # Time series data settings
    parser.add_argument('--da_strategy', type=str, default=None,
                         help='data augmentation strategy. Choose one of tf_rand/ tf_1/ ...')
    parser.add_argument('--standardize', action='store_true', default=None,
                         help='standardize the data Y/N')
    parser.add_argument('--normalize', action='store_true', default=None,
                         help='normalize the data Y/N')
    parser.add_argument('--scale_overall', action='store_true', default=None,
                         help='scale data overall train samplesY/N')
    parser.add_argument('--scale_channelwise', action='store_true', default=None,
                         help='scale per channelY/N')
    parser.add_argument('--num_workers', type=int, default=None,
                         help='parallelization of data loading, not happening for num_workers=0')
    parser.add_argument('--N', type=int, default=None,
                         help='N param for data augmentation')
    parser.add_argument('--magnitude', type=int, default=None,
                         help='magnitude param for data augmentation')
    parser.add_argument('--horizon', type=float, default=None,
                         help='horizon for the self-supervised multi task, in [0; 0.5]')
    parser.add_argument('--stride', type=float, default=None,
                         help='stride for the self-supervised multi task, in [0; 1.0]')
    parser.add_argument('--val_size', type=int, default=None, metavar='N',
                        help='size of the validation data set (default: 1000)')
    parser.add_argument('--test_size', type=int, default=None, metavar='N',
                        help='size of the test data set (default: 1000)')


    # Model specific parameters
    # Supervised
    parser.add_argument('--fully_labeled', action='store_true',
                        help='run the model fully supervised on all data')
    parser.add_argument('--mixup', type=int, default=None, metavar='MU',
                        help='mixup training Y/N aka 1/0')
    # VAT
    parser.add_argument('--xi', type=float, default=None, metavar='N',
                        help='')
    parser.add_argument('--epsilon', type=float, default=None, metavar='N',
                        help='the norm constraint for the adversarial direction (default: 1)')
    parser.add_argument('--method', type=str, default=None, metavar='N',
                        help='method for calculating the overall loss')
    parser.add_argument('--alpha', type=float, default=None, metavar='N',
                        help='controls relative balance between nll and vat loss (default: 1)/ Parametrization Mixup')
    parser.add_argument('--plot_adversarials', type=int, default=None,
                         help='plot adversarial examples')

    # MixMatch
    parser.add_argument('--K', type=int, default=None, metavar='N',
                        help='amount of unlabeled data augmentation')
    parser.add_argument('--T', type=float, default=None, metavar='N',
                        help='Temparature for sharpenig')
    parser.add_argument('--lambda_u', type=int, default=None, metavar='N',
                        help='max factor for weighting usv loss (def.: 75)')
    parser.add_argument('--plot_mixup', type=int, default=None, metavar='N',
                        help='activates the plot functionality for mixup (default: False)')

    # Mean Teacher
    parser.add_argument('--max_w', type=float, default=None, metavar='N',
                        help='max ramp up weight for the unsupervised loss')
    parser.add_argument('--alpha_ema', type=float, default=None, metavar='N',
                        help='alpha for EMA teacher update (default: 0.9)')
    parser.add_argument('--rampup_length', type=float, default=None, metavar='N',
                        help='rampup length learning rate (Mean Teacher) and lambda_u (Mixmatch)')

    # Ladder
    parser.add_argument('--loss_weights', nargs='+', type=float,
                        default=None)
    parser.add_argument('--noise_sd', type=float, default=None, metavar='N')

    # Self-supervised Multitask Model
    parser.add_argument('--lambda', type=float, default=None, metavar='N',
                        help='weighting of the forecasting auxiliary loss (def.: 1.0)')


    return parser

def update_config(args):
    """
    """
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    for name, value in vars(args).items():
        if value is None:
            continue

        for key in config.keys():
            if config[key].__contains__(name):
                config[key][name] = value

    return config