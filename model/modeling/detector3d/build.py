import pdb

from model.modeling.detector3d.detector3d import DETECTOR3D

def build_3d_detector(cfg):
    if cfg.MODEL.DETECTOR3D.DETECT_ARCHITECTURE == 'transformer':
        detector = DETECTOR3D(cfg)

    return detector