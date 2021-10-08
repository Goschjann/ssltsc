from unittest import TestCase
from torch.utils.data import DataLoader

from dl4d.datasets.cifar10 import Cifar10
from ssltsc.architectures.utils import backbone_factory
from ssltsc.models.supervised import Supervised
from ssltsc.models.ladder import LadderNet
from ssltsc.models.meanteacher import MeanTeacher
from ssltsc.models.mixmatch import MixMatch
from ssltsc.models.vat import VAT

import numpy as np
import os

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT_PATH, 'tests')


class TestImageModel(TestCase):
    path = os.path.join(PROJECT_ROOT_PATH, "tests", "cifar10/")
    n_classes = 10
    channels = 3
    height = width = 32

    batch_size = 24

    @classmethod
    def setUpClass(cls) -> None:
        cls.cifar_dataloader = DataLoader(
            dataset=Cifar10(root=DATA_ROOT, part='test'),
            batch_size=cls.batch_size)



class TestLadderNet(TestImageModel):
    backbone, backbone_dict = backbone_factory('ladder', 'cifar10',
                                               TestImageModel.n_classes,
                                               TestImageModel.channels,
                                               None)
    def test_model_predict(self):
        model = LadderNet(backbone=self.backbone, backbone_dict=self.backbone_dict)

        y_pred, _, y_true = model.predict(self.cifar_dataloader)
        self.assertEqual(y_pred.shape[0], y_true.shape[0])  # same number of predictions as ground truths
        self.assertEqual(y_pred.shape[1], max(y_true)+1)  # same number of predicted classes

    def test_model_evaluation(self):
        model = LadderNet(backbone=self.backbone, backbone_dict=self.backbone_dict)

        metrics = model.evaluate(self.cifar_dataloader)
        self.assertIn('accuracy', metrics)  # Simple sanitycheck - other metrics exists too

    def test_model_loss_calculation(self):
        rc_layers = [
            ([4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
             13),
            ([4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0], 2)]

        for loss_weights, asserted_num_rc_losses in rc_layers:
            model = LadderNet(backbone=self.backbone,
                              backbone_dict=self.backbone_dict,
                              loss_weights=loss_weights
                              )

            metrics = model.validate(step=0, hp_dict={'lr': 0.01},
                                     train_dataloader=self.cifar_dataloader,
                                     val_dataloader=self.cifar_dataloader,
                                     verbose=False)

            self.assertEqual(1, len(model.history))  # One row is added to the history
            self.assertTrue(isinstance(metrics, dict))
            self.assertTrue(any([np.isscalar(x) for x in metrics.values()]))  # All metrics are scalars
            has_rc_losses = ['train_rc_loss' in metric for metric in metrics.keys()].count(True)
            self.assertEqual(asserted_num_rc_losses, has_rc_losses)

    def test_model_train(self):
        pass



class TestSupervisedModel(TestImageModel):

    backbone, backbone_dict = backbone_factory('wideresnet28', 'cifar10',
                                               TestImageModel.n_classes,
                                               TestImageModel.channels, None)

    def test_model_predict(self):
        """
        tests predict functionality of the ABC class through a Supervised instance
        """
        model = Supervised(backbone=self.backbone, backbone_dict=self.backbone_dict)

        y_pred, _, y_true = model.predict(self.cifar_dataloader)
        self.assertEqual(y_pred.shape[0], y_true.shape[0])  # same number of predictions as ground truths
        self.assertEqual(y_pred.shape[1], max(y_true)+1)  # same number of predicted classes

    def test_model_evaluation(self):
        model = Supervised(backbone=self.backbone, backbone_dict=self.backbone_dict)
        metrics = model.evaluate(self.cifar_dataloader)
        self.assertIn('accuracy', metrics)  # Simple sanitycheck - other metrics exists too

    def test_model_validate(self):
        model = Supervised(backbone=self.backbone, backbone_dict=self.backbone_dict)

        metrics = model.validate(step=0, hp_dict={'lr': 0.01}, train_dataloader=self.cifar_dataloader, val_dataloader=self.cifar_dataloader, verbose=False)

        self.assertEqual(1, len(model.history))  # One row is added to the history
        self.assertTrue(isinstance(metrics, dict))
        self.assertTrue(any([np.isscalar(x) for x in metrics.values()]))  # All metrics are scalars


class TestMeanTeacherModel(TestImageModel):
    backbone, backbone_dict = backbone_factory('wideresnet28', 'cifar10',
                                               TestImageModel.n_classes,
                                               TestImageModel.channels, None)

    def test_model_predict(self):
        """
        tests predict functionality of the ABC class through a Supervised instance
        """
        model = MeanTeacher(backbone=self.backbone, backbone_dict=self.backbone_dict)

        # test student prediction
        y_pred, _, y_true = model.predict(self.cifar_dataloader, which='student')
        self.assertEqual(y_pred.shape[0], y_true.shape[0])  # same number of predictions as ground truths
        self.assertEqual(y_pred.shape[1], max(y_true)+1)  # same number of predicted classes

        # test teacher prediction
        y_pred, _, y_true = model.predict(self.cifar_dataloader, which='teacher')
        self.assertEqual(y_pred.shape[0], y_true.shape[0])  # same number of predictions as ground truths
        self.assertEqual(y_pred.shape[1], max(y_true)+1)  # same number of predicted classes

    def test_model_evaluation(self):
        model = MeanTeacher(backbone=self.backbone, backbone_dict=self.backbone_dict)
        metrics = model.evaluate(self.cifar_dataloader)
        self.assertIn('accuracy', metrics)  # Simple sanitycheck - other metrics exists too

    def test_model_validate(self):
        model = MeanTeacher(backbone=self.backbone, backbone_dict=self.backbone_dict)

        metrics = model.validate(step=0, hp_dict={'lr': 0.01, 'beta': 1.6}, train_dataloader=self.cifar_dataloader, val_dataloader=self.cifar_dataloader, verbose=False)

        self.assertEqual(1, len(model.history))  # One row is added to the history
        self.assertTrue(isinstance(metrics, dict))
        self.assertTrue(any([np.isscalar(x) for x in metrics.values()]))  # All metrics are scalars
        self.assertIn('train_student_loss', metrics)
        self.assertIn('train_loss', metrics)
        self.assertIn('beta', metrics)


class TestMixMatchModel(TestImageModel):
    backbone, backbone_dict = backbone_factory('wideresnet28', 'cifar10',
                                               TestImageModel.n_classes,
                                               TestImageModel.channels, None)

    def test_model_predict(self):
        """
        tests predict functionality of the ABC class through a Supervised instance
        """
        model = MixMatch(backbone=self.backbone, backbone_dict=self.backbone_dict)

        y_pred, _, y_true = model.predict(self.cifar_dataloader)
        self.assertEqual(y_pred.shape[0], y_true.shape[0])  # same number of predictions as ground truths
        self.assertEqual(y_pred.shape[1], max(y_true)+1)  # same number of predicted classes

    def test_model_evaluation(self):
        model = MixMatch(backbone=self.backbone, backbone_dict=self.backbone_dict)
        metrics = model.evaluate(self.cifar_dataloader)
        self.assertIn('accuracy', metrics)  # Simple sanitycheck - other metrics exists too

    def test_model_validate(self):
        model = MixMatch(backbone=self.backbone, backbone_dict=self.backbone_dict)

        metrics = model.validate(step=0,
                                hp_dict={'lr': 0.01, 'lambda_u': 1.6, 'train_mixmatch_loss': 0.99, 'train_l_loss': 0.8, 'train_ul_loss': 0.19},
                                train_dataloader=self.cifar_dataloader,
                                val_dataloader=self.cifar_dataloader,
                                verbose=False)

        self.assertEqual(1, len(model.history))  # One row is added to the history
        self.assertTrue(isinstance(metrics, dict))
        self.assertTrue(any([np.isscalar(x) for x in metrics.values()]))  # All metrics are scalars
        self.assertIn('train_loss', metrics)
        self.assertIn('lambda_u', metrics)


class TestVATModel(TestImageModel):
    backbone, backbone_dict = backbone_factory('wideresnet28', 'cifar10',
                                               TestImageModel.n_classes,
                                               TestImageModel.channels, None)

    def test_model_predict(self):
        """
        tests predict functionality of the ABC class through a Supervised instance
        """
        model = VAT(backbone=self.backbone, backbone_dict=self.backbone_dict)

        y_pred, _, y_true = model.predict(self.cifar_dataloader)
        self.assertEqual(y_pred.shape[0], y_true.shape[0])  # same number of predictions as ground truths
        self.assertEqual(y_pred.shape[1], max(y_true)+1)  # same number of predicted classes

    def test_model_evaluation(self):
        model = VAT(backbone=self.backbone, backbone_dict=self.backbone_dict)
        metrics = model.evaluate(self.cifar_dataloader)
        self.assertIn('accuracy', metrics)  # Simple sanitycheck - other metrics exists too

    def test_model_validate(self):
        model = VAT(backbone=self.backbone, backbone_dict=self.backbone_dict)

        metrics = model.validate(step=0,
                                 hp_dict={'lr': 0.01,
                                          'train_additional_loss': 1.6,
                                          'train_l_loss': 0.99,
                                          'train_ent_loss': 0.8,
                                          'train_vat_oss': 0.19},
                                 train_dataloader=self.cifar_dataloader,
                                 val_dataloader=self.cifar_dataloader,
                                 verbose=True)

        self.assertEqual(1, len(model.history))  # One row is added to the history
        self.assertTrue(isinstance(metrics, dict))
        self.assertTrue(any([np.isscalar(x) for x in metrics.values()]))  # All metrics are scalars
        self.assertIn('train_additional_loss', metrics)
        self.assertIn('train_ent_loss', metrics)
