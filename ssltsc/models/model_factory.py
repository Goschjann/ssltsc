from ssltsc.models.ladder import LadderNet
from ssltsc.models.supervised import Supervised
from ssltsc.models.selfsupervised import SelfSupervised
from ssltsc.models.mixmatch import MixMatch
from ssltsc.models.meanteacher import MeanTeacher
from ssltsc.models.vat import VAT
from ssltsc.models.fixmatch import Fixmatch

MODEL_DICT = {'supervised': Supervised,
              'vat': VAT,
              'mixmatch': MixMatch,
              'meanteacher': MeanTeacher,
              'ladder': LadderNet,
              'selfsupervised': SelfSupervised,
              'fixmatch': Fixmatch}


def model_factory(model_name, backbone, backbone_dict, callbacks):
    """Create a model instance

    Args:
        model_name (str): name of the model
        backbone (BaseModel): backbone class
        backbone_dict (dict): backbone dictionary to instantiate above class
        callbacks (list): list of callbacks

    Returns:
        nn.model: model instance
    """
    return MODEL_DICT[model_name](backbone=backbone,
                                  backbone_dict=backbone_dict,
                                  callbacks=callbacks)
