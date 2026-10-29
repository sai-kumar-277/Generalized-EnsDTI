from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from .dcdti import DeepConvDTI
from .dpdta import DeepDTA
from .mdprd import MDeePred
from .dp import DeepPurpose

# CPIPrediction is optional — models/cpi.py not included in repository
try:
    from .cpi import CPIPrediction
except ImportError:
    CPIPrediction = None

# from .perceivercpi import PerceiverCPI  # missing
# from .smtdta import SMTDTA              # missing

from .embedding import emb_dcdti, emb_dp, emb_dpdta, return_embedding


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()
        self.step_axis = 1

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values


def build_model(name, task, data=None):
    if name == 'dcdti':
        return DeepConvDTI(1, task)
    elif name == 'dpdta':
        return DeepDTA(85, 1200, 1, task)
    elif name == 'mdprd':
        return MDeePred(1024, 1, task)
    elif name == 'cpi':
        if CPIPrediction is None:
            raise ImportError("models/cpi.py not found in repository")
        return CPIPrediction(data.fp_num, data.word_num, 10, task)
    elif name == 'dp':
        return DeepPurpose([256+256, 1024, 1024, 512, 1], [26, 32, 64, 96], [4, 8, 12], task)
    elif name in ('perceivercpi', 'smtdta'):
        raise NotImplementedError(f"Model '{name}' is not implemented in this repository")
    else:
        raise ValueError(f"Unknown model name: {name}")


# Compatibility alias — main.py uses `declare_model`
declare_model = build_model
