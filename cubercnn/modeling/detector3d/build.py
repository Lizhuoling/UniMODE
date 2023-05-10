import pdb

from cubercnn.modeling.detector3d.petr import DETECTOR_PETR

def build_3d_detector(cfg):
    if cfg.MODEL.DETECTOR3D.DETECT_ARCHITECTURE == 'petr':
        detector = DETECTOR_PETR(cfg)

    return detector