#!/usr/bin/env python
import mlflow
import optuna
import os
import pdb
import tempfile

from optuna import Trial, samplers
from ssltsc.callbacks import TuneCallback
from ssltsc.data import load_dataloaders
from ssltsc.architectures.utils import backbone_factory
from ssltsc.experiments import update_config, get_experiment_id, save_as_yaml_file_in_mlflow, get_base_argparser, log_optuna_plots_in_mlflow, save_as_csv_file_in_mlflow, convert_to_tuner_config, convert_to_best_config
from ssltsc.models.supervised import Supervised
from ssltsc.models.vat import VAT
from ssltsc.models.meanteacher import MeanTeacher
from ssltsc.models.mixmatch import MixMatch
from ssltsc.models.model_factory import model_factory

def parse_args():
    parser = get_base_argparser(description='generic')
    parser.add_argument('--n_trials', type=int, default=None, metavar='MU',
                        help='number of trials (default: 10)')
    parser.add_argument('--time_budget', type=int, default=None, metavar='MU',
                        help='time budget in sec (default: 1000)')
    parser.add_argument('--reduction_factor', type=int, default=3, metavar='MU',
                        help='reduction factor (default: 3)')
    return parser.parse_args()

def run_experiment(args):
    # overwrite args in the config file via command line arguments
    #FIXME: only use config arguments in the following, no args.<xxx> anywhere
    configuration = update_config(args)

    # get mlflow loggers straight
    experiment_id = get_experiment_id(config=configuration)
    mlflow.start_run(experiment_id=experiment_id)

    for param, val in configuration['data_params'].items():
        mlflow.log_param(param, val)

    mlflow.set_tag(key='dataset', value=args.dataset)
    mlflow.set_tag(key='run', value='tuning')
    mlflow.set_tag(key='model', value=configuration['exp_params']['model_name'])
    save_as_yaml_file_in_mlflow(configuration, "tune_config.yaml")

    def objective(trial: Trial):
        #FIXME: pass the configuration file directly to the function
        # might be impossible because of
        print("Running trial #{}".format(trial.number))

        # sample the to be tuned params from the search space in the configuration file
        # and convert them to the optuna specific format
        tuner_config = convert_to_tuner_config(configuration, trial)

        # include params for data augmentation in tuning
        if 'stride' in tuner_config['model_params'].keys():
            tuner_config['data_params']['stride'] = tuner_config['model_params']['stride']
        if 'horizon' in tuner_config['model_params'].keys():
            tuner_config['data_params']['horizon'] = tuner_config['model_params']['horizon']

        data_dict = load_dataloaders(**tuner_config['data_params'])
        print("Data loaded")

        # Load architecture
        backbone, backbone_dict = backbone_factory(architecture=configuration['exp_params']['backbone'],
                                                   dataset=configuration['data_params']['dataset'],
                                                   n_classes=data_dict['train_data_l'].nclasses,
                                                   n_channels=data_dict['train_data_l'].nvariables,
                                                   lengthts=data_dict['train_data_l'].length,
                                                   horizon=tuner_config['model_params']['horizon'] if 'horizon' in tuner_config['model_params'].keys() else None)

        callbacks = [TuneCallback(trial)]
        model = model_factory(model_name=configuration['exp_params']['model_name'],
                              backbone=backbone,
                              backbone_dict=backbone_dict,
                              callbacks=callbacks)

        opt_dict = {'lr': tuner_config['model_params']['lr'],
                    'weight_decay': tuner_config['model_params']['weight_decay']}

        model.train(opt_dict=opt_dict, data_dict=data_dict,
                    model_params=tuner_config['model_params'],
                    exp_params=tuner_config['exp_params'])

        return model.history['val_weighted_auc'].iloc[-1]


    pruner = optuna.pruners.HyperbandPruner(
        min_resource=configuration['exp_params']['val_steps'],
        max_resource=configuration['exp_params']['n_steps'],
        reduction_factor=args.reduction_factor
    )

    # Use the objective for optuna
    study = optuna.create_study(direction='maximize',
                                pruner=pruner,
                                sampler=samplers.RandomSampler())
                                #sampler=samplers.TPESampler())
    study.optimize(objective, n_trials=args.n_trials, timeout=args.time_budget)

    # store tuning results
    df = study.trials_dataframe().sort_values(by='value', ascending=False)

    # to be sure to store the final best hpc setting
    best_config = convert_to_best_config(configuration, study.best_params)
    save_as_yaml_file_in_mlflow(best_config, 'best_' + "config.yaml")

    # store optuna history in mlflow
    save_as_csv_file_in_mlflow(data=df, filename='optuna_history.csv')

    # get the best out of three
    df = df.iloc[:3, :]

    for idx, storage_name in zip(range(3), ['bbest_', 'second_best_', 'third_best_']):
        param_names = ['params_' + param for param in [*configuration['search_space']]]
        foo = dict(df.filter(param_names).iloc[idx, :])
        hpc = {k.split('_')[1]: float(v) for k, v in foo.items()}
        best_config = convert_to_best_config(configuration, hpc)
        save_as_yaml_file_in_mlflow(best_config, storage_name + "config.yaml")

    log_optuna_plots_in_mlflow(study=study)

    mlflow.end_run()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
