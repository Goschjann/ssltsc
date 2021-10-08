#!/usr/bin/env python
import mlflow

from ssltsc.callbacks import TimerCallback

import os
import pdb
import time

from ssltsc.callbacks import ConsoleLoggingCallback, MlflowLoggingCallback, MeanTeacherConsoleLoggingCallback
from ssltsc.architectures.utils import backbone_factory
from ssltsc.data import load_dataloaders
from ssltsc.experiments import get_experiment_id, get_base_argparser, \
    update_config
from ssltsc.models.model_factory import model_factory


def parse_args():
    parser = get_base_argparser(description='generic')
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

    # DATA
    data_dict = load_dataloaders(**config['data_params'])
    # MODEL
    horizon = config['data_params']['horizon'] if "horizon" in config['data_params'] else None
    backbone, backbone_dict = backbone_factory(architecture=config['exp_params']['backbone'],
                                               dataset=config['data_params']['dataset'],
                                               horizon=horizon,
                                               n_classes=data_dict['train_data_l'].nclasses,
                                               n_channels=data_dict['train_data_l'].nvariables,
                                               lengthts=data_dict['train_data_l'].length)

    if config['exp_params']['model_name'] == 'meanteacher':
        callbacks = [MeanTeacherConsoleLoggingCallback(), MlflowLoggingCallback()]
    else:
        callbacks = [ConsoleLoggingCallback(), MlflowLoggingCallback(), TimerCallback(verbose=False)]
    model = model_factory(model_name=config['exp_params']['model_name'],
                          backbone=backbone,
                          backbone_dict=backbone_dict,
                          callbacks=callbacks)

    # log parameters (except the search space)
    for cfg in config.items():
        if cfg[0] != 'search_space':
            for k, v in cfg[1].items():
                mlflow.log_param(key=k, value=v)

    opt_dict = {'lr': config['model_params']['lr'],
                'weight_decay': config['model_params']['weight_decay']}

    # halve the labelled batch size in case num_labels < labelled batch size
    start_time = time.time()
    model.train(opt_dict=opt_dict,
                data_dict=data_dict,
                model_params=config['model_params'],
                exp_params=config['exp_params'])
    eta = round(time.time() - start_time, 3)
    print(f'Training took {eta} seconds')
    # Evaluate the model
    model.evaluate(data_loader=data_dict['test_gen'],
                   early_stopping=config['exp_params']['early_stopping'],
                   plot_reliability=False,
                   model_name=config['exp_params']['model_name'])
    mlflow.log_metric(key='es_step', value=model.es_step)

    mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args=args)
