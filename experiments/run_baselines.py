#!/usr/bin/env python
import numpy as np
import os
import mlflow
import pdb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from ssltsc import constants as c
from ssltsc.experiments import get_experiment_id, get_base_argparser, update_config
from ssltsc.models.utils import calculate_classification_metrics
from ssltsc.data import load_dataloaders

from ssltsc.models.losses import rbf_kernel_safe

def parse_args():
    parser = get_base_argparser(description='ML baseline on extracted features')
    args = parser.parse_args()
    return args


def run_experiment(args):
    config = update_config(args)

    experiment_id = get_experiment_id(config=config)
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.set_tag(key='dataset', value=args.dataset)
    mlflow.set_tag(key='run', value='training')
    mlflow.set_tag(key='model', value=config['exp_params']['model_name'])
    mlflow.log_param(key='model', value=config['exp_params']['model_name'])

    #DATA
    data_dict = load_dataloaders(**config['data_params'])
    idx_labelled_train = data_dict['train_gen_l'].batch_sampler.labelled_idxs
    idx_unlabelled_train = data_dict['train_gen_l'].batch_sampler.unlabelled_idxs

    X_train, Y_train = data_dict['train_data_l'].x[idx_labelled_train], data_dict['train_data_l'].y[idx_labelled_train]
    X_train_ul, Y_train_ul = data_dict['train_data_l'].x[idx_unlabelled_train], data_dict['train_data_l'].y[idx_unlabelled_train]
    Y_train_ul = np.full(shape=Y_train_ul.shape, fill_value=-1)
    X_test, Y_test = data_dict['test_data'].x, data_dict['test_data'].y

    # build classifier
    if config['exp_params']['model_name'] == 'randomforest':
        classifier = RandomForestClassifier(n_estimators=config['model_params']['n_estimators'],
                                            max_depth=config['model_params']['max_depth'],
                                            random_state=1)
    elif config['exp_params']['model_name'] == 'logisticregression':
        classifier = LogisticRegression(penalty=config['model_params']['penalty'],
                                        max_iter=config['model_params']['max_iter'],
                                        random_state=1)
    elif config['exp_params']['model_name'] == 'labelpropagation':
        # concat labelled/ unlabelled data
        X_train = np.concatenate([X_train, X_train_ul])
        Y_train = np.concatenate([Y_train, Y_train_ul])
        classifier = LabelPropagation(gamma=config['model_params']['gamma'],
                                      n_neighbors=config['model_params']['n_neighbors'],
                                      n_jobs=config['model_params']['n_jobs'],
                                      kernel=rbf_kernel_safe if config['model_params']['kernel'] == 'rbf_kernel_safe' else config['model_params']['kernel'],
                                      max_iter=config['model_params']['max_iter'])
    elif config['exp_params']['model_name'] == 'labelspreading':
        # concat labelled/ unlabelled data
        X_train = np.concatenate([X_train, X_train_ul])
        Y_train = np.concatenate([Y_train, Y_train_ul])
        classifier = LabelSpreading(gamma=config['model_params']['gamma'],
                                    alpha=config['model_params']['alpha'],
                                    n_neighbors=config['model_params']['n_neighbors'],
                                    n_jobs=config['model_params']['n_jobs'],
                                    kernel=rbf_kernel_safe if config['model_params']['kernel'] == 'rbf_kernel_safe' else config['model_params']['kernel'],
                                    max_iter=config['model_params']['max_iter'])

    classifier.fit(X=X_train, y=Y_train)

    yhat_prob = classifier.predict_proba(X_test)
    final_metrics = calculate_classification_metrics(yhat_prob, Y_test)

    print('Final test acc: {:.4f} w. Auc {:.4f} macro Auc {:.4f} XE {:.4f} microF1 {:.4f}'.format(
        final_metrics['accuracy'], final_metrics['weighted_auc'], final_metrics['macro_auc'],
        final_metrics['cross_entropy'], final_metrics['micro_f1']))

    # Append test to the metrics and log to mlflow
    final_metrics = {"test_"+metric: v for metric, v in final_metrics.items()}
    mlflow.log_metrics(final_metrics)

    # log parameters (except the search space)
    for cfg in config.items():
        if cfg[0] != 'search_space':
            for k, v in cfg[1].items():
                mlflow.log_param(key=k, value=v)

    mlflow.end_run()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args=args)