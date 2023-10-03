# Copyright (c) Meta Platforms, Inc. and affiliates
import pdb
import os
from detectron2.config import CfgNode as CN

def get_cfg_defaults(cfg):

    # A list of category names which will be used
    cfg.DATASETS.CATEGORY_NAMES = []
    cfg.DATASETS.DATA_RATIO = 1.0

    # The category names which will be treated as ignore
    # e.g., not counting as background during training
    # or as false positives during evaluation.
    cfg.DATASETS.IGNORE_NAMES = []

    # Should the datasets appear with the same probabilty
    # in batches (e.g., the imbalance from small and large
    # datasets will be accounted for during sampling)
    cfg.DATALOADER.BALANCE_DATASETS = False

    # The thresholds for when to treat a known box
    # as ignore based on too heavy of truncation or 
    # too low of visibility in the image. This affects
    # both training and evaluation ignores.
    cfg.DATASETS.TRUNCATION_THRES = 0.99
    cfg.DATASETS.VISIBILITY_THRES = 0.01
    cfg.DATASETS.MIN_HEIGHT_THRES = 0.00
    cfg.DATASETS.MAX_DEPTH = 1e8

    # Whether modal 2D boxes should be loaded, 
    # or if the full 3D projected boxes should be used.
    cfg.DATASETS.MODAL_2D_BOXES = False

    # Whether truncated 2D boxes should be loaded, 
    # or if the 3D full projected boxes should be used.
    cfg.DATASETS.TRUNC_2D_BOXES = True

    cfg.MODEL.STABILIZE = 0.01
    cfg.MODEL.USE_BN = True

    cfg.MODEL.DETECTOR3D = CN()
    cfg.MODEL.DETECTOR3D.DETECT_ARCHITECTURE = 'transformer'

    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR = CN()
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_FP16 = False
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.USE_NECK = True
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.NUM_QUERY = 100
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.GRID_MASK = True
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME = 'ResNet50'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.NECK_NAME = 'CPFPN'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.PTS_VOXEL_NAME = 'SPConvVoxelization'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.PTS_VOXEL_ENCODER = 'HardSimpleVFE'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.PTS_MIDDLE_ENCODER = 'SparseEncoder'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.PTS_BACKBONE = 'SECOND'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.PTS_NECK = 'SECONDFPN'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_NAME = 'PETR_TRANSFORMR'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.ITER_QUERY_UPDATE = True
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.ENC_BLOCK_X = 5
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.ENC_BLOCK_Y = 5
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.ENC_BLOCK_Z = 1
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_DROPOUT = 0.1
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.MATCHER_NAME = 'HungarianAssigner3D'
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.LLS_SPARSE = -1.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.NAIVE_DEPTH_HEAD = False

    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.VISION_FUSION_LEVEL = 4
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TEXT_FUSION_POSITION = 'after'  # before or after
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.FEAT_LEVEL_IDXS = ('stage2', 'stage3', 'stage4', 'stage5')
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.DOWNSAMPLE_FACTOR = 16

    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD = CN()
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.UNCERN_RANGE = [-10, 10]
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.CLS_WEIGHT = 2.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.LOC_WEIGHT = 0.25
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DIM_WEIGHT = 0.25
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.POSE_WEIGHT = 0.25
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DEPTH_HEAD_WEIGHT = 1.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.ENC_NUM = 0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DEC_NUM = 6
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DET2D_AREA_FACTOR = 0.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DET2D_FROM = 'GT2D' # 'GLIP' or 'GT2D'

    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.POSITION_RANGE = [-65, 65, -25, 40, 0.0, 220]
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.GRID_SIZE = [2.0, 65.0, 2.0]
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.ENC_CAM_INTRINSIC = False

    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS = CN()
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.CONFIDENCE_2D_THRE = 0.1
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.CONFIDENCE_3D_THRE = -1.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.OUTPUT_CONFIDENCE = '2D'

    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL = CN()
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL = False
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.FOCAL_DECOUPLE = False
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.VIRTUAL_FOCAL_Y = 512.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.PROPOSAL_NUMBER = 100
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.OBJ_LOSS_WEIGHT = 1.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.DEPTH_LOSS_WEIGHT = 1.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.OFFSET_LOSS_WEIGHT = 1.0
    cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.PROPOSAL_CONF_THRE = 0.3

    # sgd, adam, adam+amsgrad, adamw, adamw+amsgrad
    cfg.SOLVER.TYPE = 'sgd'
    cfg.SOLVER.VIRTUAL_EPOCHS = 20
    cfg.SOLVER.VIRTUAL_LOSS = 1.0

    cfg.MODEL.RESNETS.TORCHVISION = True
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    cfg.TEST.VISIBILITY_THRES = 1/2.0
    cfg.TEST.TRUNCATION_THRES = 1/2.0

    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.COLOR_AUG = False
    cfg.INPUT.RESIZE_TGT_SIZE = (1280, 1024)

    # weight path specifically for pretraining (no checkpointables will be loaded)
    cfg.MODEL.WEIGHTS_PRETRAIN = ''
