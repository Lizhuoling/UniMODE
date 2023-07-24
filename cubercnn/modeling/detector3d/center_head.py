import pdb

import torch
from torch import nn
import torch.nn.functional as F


class CENTER_HEAD(nn.Module):
    def __init__(cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.proposal_number = cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.PROPOSAL_NUMBER
        self.embed_dims = 256