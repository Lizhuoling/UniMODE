import pdb
import torch

from detectron2.utils.logger import _log_api_usage
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY

def build_model(cfg, priors=None):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg, priors=priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model

