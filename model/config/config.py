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

    # Threshold used for matching and filtering boxes
    # inside of ignore regions, within the RPN and ROIHeads
    cfg.MODEL.RPN.IGNORE_THRESHOLD = 0.5

    cfg.MODEL.DETECTOR3D = CN()
    cfg.MODEL.DETECTOR3D.DETECT_ARCHITECTURE = 'petr'

    cfg.MODEL.DETECTOR3D.PETR = CN()
    cfg.MODEL.DETECTOR3D.PETR.BACKBONE_FP16 = False
    cfg.MODEL.DETECTOR3D.PETR.USE_NECK = True
    cfg.MODEL.DETECTOR3D.PETR.NUM_QUERY = 100
    cfg.MODEL.DETECTOR3D.PETR.GRID_MASK = True
    cfg.MODEL.DETECTOR3D.PETR.BACKBONE_NAME = 'ResNet50'
    cfg.MODEL.DETECTOR3D.PETR.NECK_NAME = 'CPFPN'
    cfg.MODEL.DETECTOR3D.PETR.TRANSFORMER_NAME = 'PETR_TRANSFORMR'
    cfg.MODEL.DETECTOR3D.PETR.ITER_QUERY_UPDATE = True
    cfg.MODEL.DETECTOR3D.PETR.ENC_BLOCK_X = 5
    cfg.MODEL.DETECTOR3D.PETR.ENC_BLOCK_Y = 5
    cfg.MODEL.DETECTOR3D.PETR.ENC_BLOCK_Z = 1
    cfg.MODEL.DETECTOR3D.PETR.TRANSFORMER_DROPOUT = 0.1
    cfg.MODEL.DETECTOR3D.PETR.MATCHER_NAME = 'HungarianAssigner3D'
    cfg.MODEL.DETECTOR3D.PETR.LLS_SPARSE = -1.0
    cfg.MODEL.DETECTOR3D.PETR.NAIVE_DEPTH_HEAD = False

    cfg.MODEL.DETECTOR3D.PETR.VISION_FUSION_LEVEL = 4
    cfg.MODEL.DETECTOR3D.PETR.TEXT_FUSION_POSITION = 'after'  # before or after
    cfg.MODEL.DETECTOR3D.PETR.FEAT_LEVEL_IDXS = ('stage2', 'stage3', 'stage4', 'stage5')
    cfg.MODEL.DETECTOR3D.PETR.DOWNSAMPLE_FACTOR = 16
    cfg.MODEL.DETECTOR3D.PETR.LID = False
    cfg.MODEL.DETECTOR3D.PETR.LID_DEPTH = False

    cfg.MODEL.DETECTOR3D.PETR.HEAD = CN()
    cfg.MODEL.DETECTOR3D.PETR.HEAD.OV_CLS_HEAD = True
    cfg.MODEL.DETECTOR3D.PETR.HEAD.UNCERN_RANGE = [-10, 10]
    cfg.MODEL.DETECTOR3D.PETR.HEAD.CLS_WEIGHT = 2.0
    cfg.MODEL.DETECTOR3D.PETR.HEAD.LOC_WEIGHT = 0.25
    cfg.MODEL.DETECTOR3D.PETR.HEAD.DIM_WEIGHT = 0.25
    cfg.MODEL.DETECTOR3D.PETR.HEAD.POSE_WEIGHT = 0.25
    cfg.MODEL.DETECTOR3D.PETR.HEAD.DEPTH_HEAD_WEIGHT = 1.0
    cfg.MODEL.DETECTOR3D.PETR.HEAD.DET_2D_L1_WEIGHT = 5.0
    cfg.MODEL.DETECTOR3D.PETR.HEAD.DET_2D_GIOU_WEIGHT = 2.0
    cfg.MODEL.DETECTOR3D.PETR.HEAD.ENC_NUM = 0
    cfg.MODEL.DETECTOR3D.PETR.HEAD.DEC_NUM = 6
    cfg.MODEL.DETECTOR3D.PETR.HEAD.DET2D_AREA_FACTOR = 0.0
    cfg.MODEL.DETECTOR3D.PETR.HEAD.DET2D_FROM = 'GT2D' # 'GLIP' or 'GT2D'

    cfg.MODEL.DETECTOR3D.PETR.HEAD.POSITION_RANGE = [-65, 65, -25, 40, 0.0, 220]
    cfg.MODEL.DETECTOR3D.PETR.HEAD.GRID_SIZE = [2.0, 65.0, 2.0]
    cfg.MODEL.DETECTOR3D.PETR.HEAD.D_BOUND = [0.1, 230, 1.8]
    cfg.MODEL.DETECTOR3D.PETR.HEAD.ENC_CAM_INTRINSIC = False
    cfg.MODEL.DETECTOR3D.PETR.HEAD.CLS_MASK = -1.0

    cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS = CN()
    cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.CONFIDENCE_2D_THRE = 0.1
    cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.CONFIDENCE_3D_THRE = -1.0
    cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.OUTPUT_CONFIDENCE = '2D'

    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL = CN()
    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL = False
    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.FOCAL_DECOUPLE = False
    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.VIRTUAL_FOCAL_Y = 512.0
    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.PROPOSAL_NUMBER = 100
    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.OBJ_LOSS_WEIGHT = 1.0
    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.DEPTH_LOSS_WEIGHT = 1.0
    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.OFFSET_LOSS_WEIGHT = 1.0
    cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.PROPOSAL_CONF_THRE = 0.3

    # Configuration for cube head
    cfg.MODEL.ROI_CUBE_HEAD = CN()
    cfg.MODEL.ROI_CUBE_HEAD.NAME = "CubeHead"
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE = "ROIAlignV2"

    # Settings for the cube head features
    cfg.MODEL.ROI_CUBE_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_CUBE_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_CUBE_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_CUBE_HEAD.FC_DIM = 1024
    
    # the style to predict Z with currently supported
    # options --> ['direct', 'sigmoid', 'log', 'clusters']
    cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE = "direct"

    # the style to predict pose with currently supported
    # options --> ['6d', 'euler', 'quaternion']
    cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE = "6d"

    # Whether to scale all 3D losses by inverse depth
    cfg.MODEL.ROI_CUBE_HEAD.INVERSE_Z_WEIGHT = False

    # Virtual depth puts all predictions of depth into
    # a shared virtual space with a shared focal length. 
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH = True
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL = 512.0

    # If true, then all losses are computed using the 8 corners
    # such that they are all in a shared scale space. 
    # E.g., their scale correlates with their impact on 3D IoU.
    # This way no manual weights need to be set.
    cfg.MODEL.ROI_CUBE_HEAD.DISENTANGLED_LOSS = True

    # When > 1, the outputs of the 3D head will be based on
    # a 2D scale clustering, based on 2D proposal height/width.
    # This parameter describes the number of bins to cluster.
    cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS = 1

    # Whether batch norm is enabled during training. 
    # If false, all BN weights will be frozen. 
    cfg.MODEL.USE_BN = True

    # Whether to predict the pose in allocentric space. 
    # The allocentric space may correlate better with 2D 
    # images compared to egocentric poses. 
    cfg.MODEL.ROI_CUBE_HEAD.ALLOCENTRIC_POSE = True

    # Whether to use chamfer distance for disentangled losses
    # of pose. This avoids periodic issues of rotation but 
    # may prevent the pose "direction" from being interpretable.
    cfg.MODEL.ROI_CUBE_HEAD.CHAMFER_POSE = True

    # Should the prediction heads share FC features or not. 
    # These include groups of uv, z, whl, pose.
    cfg.MODEL.ROI_CUBE_HEAD.SHARED_FC = True

    # Check for stable gradients. When inf is detected, skip the update. 
    # This prevents an occasional bad sample from exploding the model. 
    # The threshold below is the allows percent of bad samples. 
    # 0.0 is off, and 0.01 is recommended for minor robustness to exploding.
    cfg.MODEL.STABILIZE = 0.01
    
    # Whether or not to use the dimension priors
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED = True

    # How prior dimensions should be computed? 
    # The supported modes are ["exp", "sigmoid"]
    # where exp is unbounded and sigmoid is bounded
    # between +- 3 standard deviations from the mean.
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_FUNC = 'exp'

    # weight for confidence loss. 0 is off.
    cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE = 1.0

    # Loss weights for XY, Z, Dims, Pose
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_XY = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_Z = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_POSE = 1.0

    cfg.MODEL.DLA = CN()

    # Supported types for DLA backbones are...
    # dla34, dla46_c, dla46x_c, dla60x_c, dla60, dla60x, dla102x, dla102x2, dla169
    cfg.MODEL.DLA.TYPE = 'dla34'

    # Only available for dla34, dla60, dla102
    cfg.MODEL.DLA.TRICKS = False

    # A joint loss for the disentangled loss.
    # All predictions are computed using a corner
    # or chamfers loss depending on chamfer_pose!
    # Recommened to keep this weight small: [0.05, 0.5]
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT = 1.0

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

    # When True, we will use localization uncertainty
    # as the new IoUness score in the RPN.
    cfg.MODEL.RPN.OBJECTNESS_UNCERTAINTY = 'IoUness'

    # If > 0.0 this is the scaling factor that will be applied to
    # an RoI 2D box before doing any pooling to give more context. 
    # Ex. 1.5 makes width and height 50% larger. 
    cfg.MODEL.ROI_CUBE_HEAD.SCALE_ROI_BOXES = 0.0

    # weight path specifically for pretraining (no checkpointables will be loaded)
    cfg.MODEL.WEIGHTS_PRETRAIN = ''
