#!/usr/bin/env python
import mlflow
import optuna
import os
import pdb
import tempfile
import numpy as np

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
from ssltsc.models.utils import calculate_classification_metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from ssltsc.models.losses import rbf_kernel_safe

def parse_args():
    parser = get_base_argparser(description='ml model tuner')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='amound of random search model evals (default: 10)')
    return parser.parse_args()

def run_experiment(args):
    # overwrite args in the config file via command line arguments
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

    data_dict = load_dataloaders(**configuration['data_params'])
    print("Data loaded")

    def objective(trial: Trial):
        print("Running trial #{}".format(trial.number))

        #DATA
        idx_labelled_train = data_dict['train_gen_l'].batch_sampler.labelled_idxs
        idx_unlabelled_train = data_dict['train_gen_l'].batch_sampler.unlabelled_idxs

        X_train, Y_train = data_dict['train_data_l'].x[idx_labelled_train], data_dict['train_data_l'].y[idx_labelled_train]
        X_train_ul, Y_train_ul = data_dict['train_data_l'].x[idx_unlabelled_train], data_dict['train_data_l'].y[idx_unlabelled_train]
        Y_train_ul = np.full(shape=Y_train_ul.shape, fill_value=-1)
        X_val, Y_val = data_dict['val_data'].x, data_dict['val_data'].y


        # update config and suggest hyperpars
        tuner_config = convert_to_tuner_config(configuration, trial)
        if tuner_config['exp_params']['model_name'] == 'randomforest':
            classifier = RandomForestClassifier(n_estimators=tuner_config['model_params']['n_estimators'],
                                                max_depth=tuner_config['model_params']['max_depth'],
                                                random_state=1)
        elif tuner_config['exp_params']['model_name'] == 'logisticregression':
            classifier = LogisticRegression(penalty=tuner_config['model_params']['penalty'],
                                            # solver='liblinear',
                                            max_iter=tuner_config['model_params']['max_iter'],
                                            random_state=1)
        elif tuner_config['exp_params']['model_name'] == 'labelpropagation':
            # concat labelled/ unlabelled data
            X_train = np.concatenate([X_train, X_train_ul])
            Y_train = np.concatenate([Y_train, Y_train_ul])
            classifier = LabelPropagation(gamma=tuner_config['model_params']['gamma'],
                                        n_neighbors=tuner_config['model_params']['n_neighbors'],
                                        n_jobs=tuner_config['model_params']['n_jobs'],
                                        kernel=rbf_kernel_safe,
                                        max_iter=tuner_config['model_params']['max_iter'])
        elif tuner_config['exp_params']['model_name'] == 'labelspreading':
            # concat labelled/ unlabelled data
            X_train = np.concatenate([X_train, X_train_ul])
            Y_train = np.concatenate([Y_train, Y_train_ul])
            classifier = LabelSpreading(gamma=tuner_config['model_params']['gamma'],
                                        alpha=tuner_config['model_params']['alpha'],
                                        n_neighbors=tuner_config['model_params']['n_neighbors'],
                                        n_jobs=tuner_config['model_params']['n_jobs'],
                                        kernel=rbf_kernel_safe if tuner_config['model_params']['kernel'] == 'rbf_kernel_safe' else tuner_config['model_params']['kernel'],
                                        max_iter=tuner_config['model_params']['max_iter'])
        classifier.fit(X=X_train, y=Y_train)

        # validate
        Y_hat = classifier.predict_proba(X=X_val)
        metrics = calculate_classification_metrics(Y_hat, Y_val)

        return metrics['weighted_auc']

    # Use the objective for optuna
    study = optuna.create_study(direction='maximize',
                                #pruner=pruner,
                                sampler=samplers.RandomSampler())
    study.optimize(objective, n_trials=args.n_trials)

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
        # pdb.set_trace()
        # hpc = {k.split('_')[1]: float(v) for k, v in foo.items()}
        hpc = {k.split('_')[1]: v for k, v in foo.items()}
        best_config = convert_to_best_config(configuration, hpc)
        save_as_yaml_file_in_mlflow(best_config, storage_name + "config.yaml")

    log_optuna_plots_in_mlflow(study=study)

    mlflow.end_run()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
