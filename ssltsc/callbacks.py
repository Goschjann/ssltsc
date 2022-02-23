import os
import tempfile
from statistics import mean

import optuna
import mlflow
import time
import pdb
from optuna import Trial
from abc import ABC
from pandas import DataFrame


class Callback(ABC):
    """
    Base callback class. The method naming convention is inspired by the
    Keras naming convention.
    https://keras.io/guides/writing_your_own_callbacks/
    """
    def on_train_batch_start(self):
        pass

    def on_train_batch_end(self, step: int):
        pass

    def on_validation_end(self, step: int, metrics: dict):
        """
        Will be called every time in the train loop when the validation step is over.
        Args:
            step:
            metrics:
        Returns:
        """
        pass

    def on_evaluation_end(self, metrics: dict):
        """
        Will be called when an evaluation has been done.
        Args:
            metrics:
        """
        pass

    def on_train_end(self, history: DataFrame):
        pass


class ConsoleLoggingCallback(Callback):
    def on_validation_end(self, step: int, metrics: dict):
        validate_string = 'step {} | TRAIN: loss {:0.3f} acc {:0.3f} auc {:0.3f} VAL: loss {:0.3f} acc {:0.3f} auc {:0.3f}'
        assert any([bool(metric in metrics) for metric in ['train_loss', 'train_accuracy', 'train_weighted_auc', 'val_loss', 'val_accuracy', 'val_weighted_auc']]), "Missing metric for logging to console."

        print(validate_string.format(step, metrics['train_cross_entropy'], metrics['train_accuracy'], metrics['train_weighted_auc'], metrics['val_cross_entropy'], metrics['val_accuracy'], metrics['val_weighted_auc']))

    def on_evaluation_end(self, metrics: dict):
        print('Final test acc: {:.4f} w. Auc {:.4f} macro Auc {:.4f} XE {:.4f} microF1 {:.4f}'.format(
            metrics['accuracy'], metrics['weighted_auc'], metrics['macro_auc'],
            metrics['cross_entropy'], metrics['micro_f1']))

class MeanTeacherConsoleLoggingCallback(Callback):
    """
    This logger callback is necessary as the Mean Teacher model both has a student and a teacher loss
    """
    def on_validation_end(self, step: int, metrics: dict) -> bool:
        validate_string = 'step {} | TRAIN: student loss {:0.3f} acc {:0.3f} auc {:0.3f}, ' \
                          'teacher loss {:0.3f} acc {:0.3f} auc {:0.3f}, \n      | VAL: student loss {:0.3f} ' \
                          'acc {:0.3f} auc {:0.3f}, teacher loss {:0.3f} acc {:0.3f} auc {:0.3f}'

        print(validate_string.format(step,
                                     metrics['train_student_loss'],
                                     metrics['train_student_accuracy'],
                                     metrics['train_student_weighted_auc'],
                                     metrics['train_loss'],
                                     metrics['train_accuracy'],
                                     metrics['train_weighted_auc'],
                                     metrics['val_student_loss'],
                                     metrics['val_student_accuracy'],
                                     metrics['val_student_weighted_auc'],
                                     metrics['val_loss'],
                                     metrics['val_accuracy'],
                                     metrics['val_weighted_auc']))

        stop_training = False
        return stop_training


class MlflowLoggingCallback(Callback):
    def on_validation_end(self, step: int, metrics: dict):
        mlflow.log_metrics(metrics, step=step)

    def on_evaluation_end(self, metrics: dict):
        # Append test to the metrics and log to mlflow
        final_metrics = {"test_" + metric: v for metric, v in metrics.items()}
        mlflow.log_metrics(final_metrics)

    # def on_train_end(self, history: DataFrame):

    #     # We log the history of the model as a csv artifact in mlflow
    #     # with tempfile.TemporaryDirectory() as tmp_dir:
    #     #     history_path = os.path.join(tmp_dir, "history.csv")
    #     #     history.to_csv(history_path)
    #     #     mlflow.log_artifact(history_path)


class TimerCallback(Callback):
    def __init__(self, verbose=False, log_to_mlflow=True):
        self.verbose = verbose
        self.batch_times = []
        self.log_to_ml_flow = log_to_mlflow

    def on_train_batch_start(self):
        self.start = time.time()

    def on_train_batch_end(self, step: int):
        batch_time = time.time() - self.start
        self.batch_times.append(batch_time)

        if self.verbose:
            print("step {} | Batch took {:.2}s".format(step, batch_time))

    def on_validation_end(self, step: int, metrics: dict):
        if len(self.batch_times) > 0:
            avg_batch_times = mean(self.batch_times)

            if self.log_to_ml_flow:
                mlflow.log_metric('avg_train_batch_times',
                                  avg_batch_times, step=step)
            print("step - | Avg. train batch time: {:.2}s".format(avg_batch_times))
            self.batch_times = []

class TuneCallback(Callback):
    def __init__(self, trial: Trial, tuning_criterion: str):
        self.trial = trial
        self.tuning_criterion = tuning_criterion

    def on_validation_end(self, step: int, metrics: dict):
        assert self.tuning_criterion in metrics.keys(), 'Your tuning criterion is not part of the tracked performance metrics'
        value = metrics[self.tuning_criterion]
        print(f'Trial {self.trial._trial_id} step {step} with value {value}')
        self.trial.report(metrics[self.tuning_criterion], step)

        if self.trial.should_prune():
            raise optuna.TrialPruned()