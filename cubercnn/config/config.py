# Copyright (c) Meta Platforms, Inc. and affiliates
import pdb
import os
from detectron2.config import CfgNode as CN

def get_cfg_defaults(cfg):

    # A list of category names which will be used
    cfg.DATASETS.CATEGORY_NAMES = []

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
    cfg.MODEL.DETECTOR3D.PETR.NUM_QUERY = 100
    cfg.MODEL.DETECTOR3D.PETR.GRID_MASK = True
    cfg.MODEL.DETECTOR3D.PETR.BACKBONE_NAME = 'ResNet50'
    cfg.MODEL.DETECTOR3D.PETR.NECK_NAME = 'CPFPN'
    cfg.MODEL.DETECTOR3D.PETR.TRANSFORMER_NAME = 'PETR_TRANSFORMR'
    cfg.MODEL.DETECTOR3D.PETR.MATCHER_NAME = 'HungarianAssigner3D'
    cfg.MODEL.DETECTOR3D.PETR.DEPTH_LID = True

    cfg.MODEL.DETECTOR3D.PETR.HEAD = CN()
    cfg.MODEL.DETECTOR3D.PETR.HEAD.UNCERN_RANGE = [-10, 10]
    cfg.MODEL.DETECTOR3D.PETR.HEAD.CLS_WEIGHT = 2.0
    cfg.MODEL.DETECTOR3D.PETR.HEAD.REG_WEIGHT = 0.25
    cfg.MODEL.DETECTOR3D.PETR.HEAD.IOU_WEIGHT = 0.0

    cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS = CN()
    cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.CONFIDENCE_2D_THRE = 0.1
    cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.CONFIDENCE_3D_THRE = -1.0
    cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.OUTPUT_CONFIDENCE = '2D'

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

    cfg.MODEL.RESNETS.TORCHVISION = True
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    cfg.TEST.VISIBILITY_THRES = 1/2.0
    cfg.TEST.TRUNCATION_THRES = 1/2.0

    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # When True, we will use localization uncertainty
    # as the new IoUness score in the RPN.
    cfg.MODEL.RPN.OBJECTNESS_UNCERTAINTY = 'IoUness'

    # If > 0.0 this is the scaling factor that will be applied to
    # an RoI 2D box before doing any pooling to give more context. 
    # Ex. 1.5 makes width and height 50% larger. 
    cfg.MODEL.ROI_CUBE_HEAD.SCALE_ROI_BOXES = 0.0

    # weight path specifically for pretraining (no checkpointables will be loaded)
    cfg.MODEL.WEIGHTS_PRETRAIN = ''

    ### GLIP model ###
    cfg.MODEL.GLIP_MODEL = CN()
    cfg.MODEL.GLIP_MODEL.USE_GLIP = False
    cfg.MODEL.GLIP_MODEL.GLIP_INITIALIZE_QUERY = True
    cfg.MODEL.GLIP_MODEL.NO_GRADIENT = True
    cfg.MODEL.GLIP_MODEL.GLIP_WEIGHT = ""
    
    cfg.MODEL.GLIP_MODEL.MODEL = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.RPN_ONLY = False
    cfg.MODEL.GLIP_MODEL.MODEL.BOX_ON = True
    cfg.MODEL.GLIP_MODEL.MODEL.MASK_ON = False
    cfg.MODEL.GLIP_MODEL.MODEL.KEYPOINT_ON = False
    cfg.MODEL.GLIP_MODEL.MODEL.DEVICE = "cuda"

    cfg.MODEL.GLIP_MODEL.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

    cfg.MODEL.GLIP_MODEL.MODEL.RPN_ARCHITECTURE = "RPN"
    cfg.MODEL.GLIP_MODEL.MODEL.DEBUG = False  # add debug flag
    cfg.MODEL.GLIP_MODEL.MODEL.ONNX = False  # add onnx flag

    # If the WEIGHT starts with a catalog://, like :R-50, the code will look for
    # the path in paths_catalog. Else, it will use it as the specified absolute
    # path
    cfg.MODEL.GLIP_MODEL.MODEL.WEIGHT = ""
    cfg.MODEL.GLIP_MODEL.MODEL.PRETRAIN_NAME = ""

    # If LINEAR_PROB = True, only the last linear layers in rpn and roi_head are trainable
    cfg.MODEL.GLIP_MODEL.MODEL.LINEAR_PROB = False

    # -----------------------------------------------------------------------------
    # Multitask Training / Test specific parameters
    # -----------------------------------------------------------------------------
    cfg.MODEL.GLIP_MODEL.MODEL.MULTITASK = CN(new_allowed=True)

    # -----------------------------------------------------------------------------
    # INPUT
    # -----------------------------------------------------------------------------
    cfg.MODEL.GLIP_MODEL.INPUT = CN()
    # Size of the smallest side of the image during training
    cfg.MODEL.GLIP_MODEL.INPUT.MIN_SIZE_TRAIN = 800  # (800,)
    # Maximum size of the side of the image during training
    cfg.MODEL.GLIP_MODEL.INPUT.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing
    cfg.MODEL.GLIP_MODEL.INPUT.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    cfg.MODEL.GLIP_MODEL.INPUT.MAX_SIZE_TEST = 1333
    # Values to be used for image normalization
    cfg.MODEL.GLIP_MODEL.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    # Values to be used for image normalization
    cfg.MODEL.GLIP_MODEL.INPUT.PIXEL_STD = [1., 1., 1.]
    # Convert image to BGR format (for Caffe2 models), in range 0-255
    cfg.MODEL.GLIP_MODEL.INPUT.TO_BGR255 = True
    cfg.MODEL.GLIP_MODEL.INPUT.FORMAT = ''
    cfg.MODEL.GLIP_MODEL.INPUT.FIX_RES = False

    # -----------------------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------------------
    cfg.MODEL.GLIP_MODEL.AUGMENT = CN()
    cfg.MODEL.GLIP_MODEL.AUGMENT.USE_RA = 0
    cfg.MODEL.GLIP_MODEL.AUGMENT.FLIP_PROB_TRAIN = 0.5
    cfg.MODEL.GLIP_MODEL.AUGMENT.VERTICAL_FLIP_PROB_TRAIN = 0.0
    cfg.MODEL.GLIP_MODEL.AUGMENT.MULT_MIN_SIZE_TRAIN = ()

    cfg.MODEL.GLIP_MODEL.AUGMENT.BRIGHTNESS = 0.0
    cfg.MODEL.GLIP_MODEL.AUGMENT.CONTRAST = 0.0
    cfg.MODEL.GLIP_MODEL.AUGMENT.SATURATION = 0.0
    cfg.MODEL.GLIP_MODEL.AUGMENT.HUE = 0.0

    cfg.MODEL.GLIP_MODEL.AUGMENT.CROP_PROB = 0.5
    cfg.MODEL.GLIP_MODEL.AUGMENT.CROP_MIN_IOUS = (0.1, 0.3, 0.5, 0.7, 0.9)
    cfg.MODEL.GLIP_MODEL.AUGMENT.CROP_MIN_SIZE = 0.3

    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    cfg.MODEL.GLIP_MODEL.DATASETS = CN()
    # List of the dataset names for training, as present in paths_catalog.py
    cfg.MODEL.GLIP_MODEL.DATASETS.TRAIN = ()
    # List of the dataset names for testing, as present in paths_catalog.py
    cfg.MODEL.GLIP_MODEL.DATASETS.TEST = ()
    # Use is_crowd label
    cfg.MODEL.GLIP_MODEL.DATASETS.USE_CROWD = False
    cfg.MODEL.GLIP_MODEL.DATASETS.CLASS_AGNOSTIC = False
    cfg.MODEL.GLIP_MODEL.DATASETS.CLASS_CONCAT = False
    cfg.MODEL.GLIP_MODEL.DATASETS.MAX_BOX = -1
    cfg.MODEL.GLIP_MODEL.DATASETS.SAMPLE_RATIO = 0.0
    cfg.MODEL.GLIP_MODEL.DATASETS.FEW_SHOT = 0
    # SHUFFLE_SEED != 0 means shuffle the dataset in the few shot setting
    cfg.MODEL.GLIP_MODEL.DATASETS.SHUFFLE_SEED = 0
    cfg.MODEL.GLIP_MODEL.DATASETS.PREDEFINED_TEXT = ''
    cfg.MODEL.GLIP_MODEL.DATASETS.ALTERNATIVE_TRAINING = False
    cfg.MODEL.GLIP_MODEL.DATASETS.MULTISTAGE_TRAINING = False
    cfg.MODEL.GLIP_MODEL.DATASETS.REGISTER = CN(new_allowed=True)
    cfg.MODEL.GLIP_MODEL.DATASETS.BOX_THRESHOLD = 0.1
    # Duplicate Dataset
    cfg.MODEL.GLIP_MODEL.DATASETS.COCO_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.LVIS_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.FLICKR_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.MIXED_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.OBJECT365_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.VG_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.OI_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.IN_COPY = 1

    # Duplicate Dataset
    cfg.MODEL.GLIP_MODEL.DATASETS.COCO_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.FLICKR_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.MIXED_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.OBJECT365_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.VG_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.OI_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.IN_COPY = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.GENERAL_COPY = -1
    cfg.MODEL.GLIP_MODEL.DATASETS.GENERAL_COPY_TEST = -1

    # OD to Grounding
    cfg.MODEL.GLIP_MODEL.DATASETS.RANDOM_SAMPLE_NEG = -1
    cfg.MODEL.GLIP_MODEL.DATASETS.ADD_DET_PROMPT = False
    cfg.MODEL.GLIP_MODEL.DATASETS.ADD_DET_PROMPT_ADVANCED = False
    cfg.MODEL.GLIP_MODEL.DATASETS.USE_OD_AUG = False
    cfg.MODEL.GLIP_MODEL.DATASETS.USE_COCO_FORMAT = False
    cfg.MODEL.GLIP_MODEL.DATASETS.CONTROL_PROB = ()
    cfg.MODEL.GLIP_MODEL.DATASETS.DISABLE_SHUFFLE = False
    cfg.MODEL.GLIP_MODEL.DATASETS.PROMPT_VERSION = ""
    cfg.MODEL.GLIP_MODEL.DATASETS.PROMPT_LIMIT_NEG = -1
    cfg.MODEL.GLIP_MODEL.DATASETS.POS_QUESTION_PROB = 0.6
    cfg.MODEL.GLIP_MODEL.DATASETS.NEG_QUESTION_PROB = 0.8
    cfg.MODEL.GLIP_MODEL.DATASETS.FULL_QUESTION_PROB = 0.5
    cfg.MODEL.GLIP_MODEL.DATASETS.ONE_HOT = False
    cfg.MODEL.GLIP_MODEL.DATASETS.NO_MINUS_ONE_FOR_ONE_HOT = False

    cfg.MODEL.GLIP_MODEL.DATASETS.DISABLE_CLIP_TO_IMAGE = False
    cfg.MODEL.GLIP_MODEL.DATASETS.SEPARATION_TOKENS = " "

    # LVIS
    cfg.MODEL.GLIP_MODEL.DATASETS.LVIS_USE_NORMAL_AP = False
    cfg.MODEL.GLIP_MODEL.DATASETS.SPECIAL_SAFEGUARD_FOR_COCO_GROUNDING = False

    # Caption
    cfg.MODEL.GLIP_MODEL.DATASETS.BING_INDEX_LIST = []
    cfg.MODEL.GLIP_MODEL.DATASETS.CAPTION_MIN_BOX = 1
    cfg.MODEL.GLIP_MODEL.DATASETS.REPLACE_CLEAN_LABEL = False
    cfg.MODEL.GLIP_MODEL.DATASETS.FURTHER_SCREEN = False
    cfg.MODEL.GLIP_MODEL.DATASETS.CAPTION_CONF = 0.9
    cfg.MODEL.GLIP_MODEL.DATASETS.CAPTION_NMS = 0.9
    cfg.MODEL.GLIP_MODEL.DATASETS.PACK_RANDOM_CAPTION_NUMBER = 0
    cfg.MODEL.GLIP_MODEL.DATASETS.INFERENCE_CAPTION = False
    cfg.MODEL.GLIP_MODEL.DATASETS.SAMPLE_NEGATIVE_FOR_GROUNDING_DATA = -1.0
    cfg.MODEL.GLIP_MODEL.DATASETS.RANDOM_PACK_PROB = -1.0
    cfg.MODEL.GLIP_MODEL.DATASETS.NO_RANDOM_PACK_PROBABILITY = 0.0
    cfg.MODEL.GLIP_MODEL.DATASETS.SAFEGUARD_POSITIVE_CAPTION = True
    cfg.MODEL.GLIP_MODEL.DATASETS.CAPTION_FORMAT_VERSION = "v1"
    cfg.MODEL.GLIP_MODEL.DATASETS.LOCAL_DEBUG = False


    # Od in the wild
    cfg.MODEL.GLIP_MODEL.DATASETS.PREDEFINED_TEXT = None
    cfg.MODEL.GLIP_MODEL.DATASETS.TRAIN_DATASETNAME_SUFFIX = ""
    cfg.MODEL.GLIP_MODEL.DATASETS.TEST_DATASETNAME_SUFFIX = ""
    cfg.MODEL.GLIP_MODEL.DATASETS.OVERRIDE_CATEGORY = None
    cfg.MODEL.GLIP_MODEL.DATASETS.USE_OVERRIDE_CATEGORY = False
    cfg.MODEL.GLIP_MODEL.DATASETS.SUPRESS_QUERY = None
    cfg.MODEL.GLIP_MODEL.DATASETS.USE_SUPRESS_QUERY = False
    cfg.MODEL.GLIP_MODEL.DATASETS.USE_CAPTION_PROMPT = False
    cfg.MODEL.GLIP_MODEL.DATASETS.CAPTION_PROMPT = None

    cfg.MODEL.GLIP_MODEL.DATASETS.FLICKR_GT_TYPE = "separate"

    # VQA
    cfg.MODEL.GLIP_MODEL.DATASETS.DIVER_BOX_FOR_VQA = False
    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    cfg.MODEL.GLIP_MODEL.DATALOADER = CN()
    # Number of data loading threads
    cfg.MODEL.GLIP_MODEL.DATALOADER.NUM_WORKERS = 4
    # If > 0, this enforces that each collated batch should have a size divisible
    # by SIZE_DIVISIBILITY
    cfg.MODEL.GLIP_MODEL.DATALOADER.SIZE_DIVISIBILITY = 0
    # If True, each batch should contain only images for which the aspect ratio
    # is compatible. This groups portrait images together, and landscape images
    # are not batched with portrait images.
    cfg.MODEL.GLIP_MODEL.DATALOADER.ASPECT_RATIO_GROUPING = True
    # Define min number of keypoints required from GT, for example 10 out of 17
    cfg.MODEL.GLIP_MODEL.DATALOADER.MIN_KPS_PER_IMS = 0
    # Use random sampler during training
    cfg.MODEL.GLIP_MODEL.DATALOADER.USE_RANDOM_SEED = False

    cfg.MODEL.GLIP_MODEL.DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE = False
    # ---------------------------------------------------------------------------- #
    # Backbone options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE = CN()

    # The backbone conv body to use
    # The string must match a function that is imported in modeling.model_builder
    # (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
    # backbone)
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

    # Add StopGrad at a specified stage so the bottom layers are frozen
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.FREEZE = False
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.GROUP = 1
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.OUT_CHANNELS = 256 * 4
    # Option to reset bn running statics
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.RESET_BN = False
    # Backbone Normalization Level
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.NORM_LEVEL = 3
    # BN for backbone
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_BN = False
    # Sync BN for backbone
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_SYNCBN = False
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_NSYNCBN = False
    # GN for backbone
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_GN = False
    # Evo Norm for backbone
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_EN = False
    # Layers for backbone
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_DFCONV = False
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_DYRELU = False
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_SE = False
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.LAYER_SETUP = (3, 4, 6, 3)
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.LAYER_SEARCH = CN(new_allowed=True)
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.OUT_FEATURES = ("stage2", "stage3", "stage4", "stage5")
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.FPN_LAYER = ()
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.USE_CHECKPOINT = False
    # Add JF efficient det cfgs
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.EFFICIENT_DET_START_FROM = 3
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.EFFICIENT_DET_COMPOUND = 0
    cfg.MODEL.GLIP_MODEL.MODEL.BACKBONE.EFFICIENT_DET_BIFPN_VERSION = 0

    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.WEIGHT = ""
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.FREEZE = False
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT = False
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE = "bert-base-uncased"
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.LANG_DIM = 768
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN = 256
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.N_LAYERS = 1
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.UNUSED_TOKEN = 106
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False

    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.RNN_TYPE = "lstm"
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.VARIABLE_LENGTH = True
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.WORD_EMBEDDING_SIZE = 512
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.WORD_VEC_SIZE = 512
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.HIDDEN_SIZE = 512
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.BIDIRECTIONAL = True
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.INPUT_DROPOUT_P = 0.5
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.DROPOUT_P = 0.2
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.CORPUS_PATH = ""
    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.VOCAB_SIZE = 0

    cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.PAD_MAX = True
    # ---------------------------------------------------------------------------- #
    # FPN options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.FPN = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.FREEZE = False
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.USE_GN = False
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.USE_RELU = False
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.USE_DYRELU = False
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.DROP_BLOCK = True
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.DROP_PROB = 0.3
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.DROP_SIZE = 3
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.USE_SPP = False
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.USE_PAN = False
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.USE_DYHEAD = False
    cfg.MODEL.GLIP_MODEL.MODEL.FPN.RETURN_SWINT_FEATURE_BEFORE_FUSION = False
    # ---------------------------------------------------------------------------- #
    # BIFPN options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.BIFPN = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.BIFPN.NUM_REPEATS = 1
    cfg.MODEL.GLIP_MODEL.MODEL.BIFPN.USE_ATTENTION = True

    # ---------------------------------------------------------------------------- #
    # Group Norm options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.GROUP_NORM = CN()
    # Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
    cfg.MODEL.GLIP_MODEL.MODEL.GROUP_NORM.DIM_PER_GP = -1
    # Number of groups in GroupNorm (-1 if using DIM_PER_GP)
    cfg.MODEL.GLIP_MODEL.MODEL.GROUP_NORM.NUM_GROUPS = 16
    # GroupNorm's small constant in the denominator
    cfg.MODEL.GLIP_MODEL.MODEL.GROUP_NORM.EPSILON = 1e-5

    # ---------------------------------------------------------------------------- #
    # Evo Norm options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.EVO_NORM = CN()
    # Number of groups in EvoNorm (-1 if using DIM_PER_GP)
    cfg.MODEL.GLIP_MODEL.MODEL.EVO_NORM.NUM_GROUPS = 8
    # EvoNorm's small constant in the denominator
    cfg.MODEL.GLIP_MODEL.MODEL.EVO_NORM.EPSILON = 1e-5

    # ---------------------------------------------------------------------------- #
    # RetinaNet Options (Follow the Detectron version)
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.RETINANET = CN()
    # This is the number of foreground classes and background.
    cfg.MODEL.GLIP_MODEL.MODEL.RETINANET.NUM_CLASSES = 81
    # Convolutions to use in the cls and bbox tower
    # NOTE: this doesn't include the last conv for logits
    cfg.MODEL.GLIP_MODEL.MODEL.RETINANET.NUM_CONVS = 4
    # During inference, #locs to select based on cls score before NMS is performed
    # per FPN level
    cfg.MODEL.GLIP_MODEL.MODEL.RETINANET.PRE_NMS_TOP_N = 1000
    # Prior prob for the positives at the beginning of training. This is used to set
    # the bias init for the logits layer
    cfg.MODEL.GLIP_MODEL.MODEL.RETINANET.PRIOR_PROB = 0.01
    # Inference cls score threshold, anchors with score > INFERENCE_TH are
    # considered for inference
    cfg.MODEL.GLIP_MODEL.MODEL.RETINANET.INFERENCE_TH = 0.05
    # NMS threshold used in RetinaNet
    cfg.MODEL.GLIP_MODEL.MODEL.RETINANET.NMS_TH = 0.4
    cfg.MODEL.GLIP_MODEL.MODEL.RETINANET.DETECTIONS_PER_IMG = 100

    # ---------------------------------------------------------------------------- #
    # Focal Loss Options (Follow the Detectron version)
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.FOCAL = CN()
    # Weight for bbox_regression loss
    cfg.MODEL.GLIP_MODEL.MODEL.FOCAL.BBOX_REG_WEIGHT = 4.0
    # Smooth L1 loss beta for bbox regression
    cfg.MODEL.GLIP_MODEL.MODEL.FOCAL.BBOX_REG_BETA = 0.11
    # IoU overlap ratio for labeling an anchor as positive
    # Anchors with >= iou overlap are labeled positive
    cfg.MODEL.GLIP_MODEL.MODEL.FOCAL.FG_IOU_THRESHOLD = 0.5
    # IoU overlap ratio for labeling an anchor as negative
    # Anchors with < iou overlap are labeled negative
    cfg.MODEL.GLIP_MODEL.MODEL.FOCAL.BG_IOU_THRESHOLD = 0.4
    # Focal loss parameter: alpha
    cfg.MODEL.GLIP_MODEL.MODEL.FOCAL.LOSS_ALPHA = 0.25
    # Focal loss parameter: gamma
    cfg.MODEL.GLIP_MODEL.MODEL.FOCAL.LOSS_GAMMA = 2.0

    # ---------------------------------------------------------------------------- #
    # FCOS Options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.PRIOR_PROB = 0.01
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.INFERENCE_TH = 0.05
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.NMS_TH = 0.6
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.PRE_NMS_TOP_N = 1000

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.NUM_CONVS = 4
    # if use deformable conv to align features
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.USE_DFCONV = False

    # if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 0.0
    # IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.IOU_LOSS_TYPE = "iou"

    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.NORM_REG_TARGETS = False
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.CENTERNESS_ON_REG = False
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.USE_GT_CENTER = False

    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.DETECTIONS_PER_IMG = 100
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.USE_GN = False
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.USE_BN = False

    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.0
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.PRE_NMS_TOP_N_TRAIN = 3000
    cfg.MODEL.GLIP_MODEL.MODEL.FCOS.POST_NMS_TOP_N_TRAIN = 1000

    # ---------------------------------------------------------------------------- #
    # ATSS Options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.NUM_CLASSES = 81  # the number of classes including background
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.PRIOR_PROB = 0.01
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.INFERENCE_TH = 0.05
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.NMS_TH = 0.6
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.PRE_NMS_TOP_N = 1000

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.NUM_CONVS = 4
    # the channels of convolutions used in the cls and bbox tower
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.CHANNELS = 128
    # if use deformable conv to align features
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.USE_DFCONV = False

    # topk for selecting candidate positive samples from each level
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.TOPK = 9

    # Weight for bbox_regression loss
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.DETECTIONS_PER_IMG = 100
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.USE_GN = False
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.USE_BN = False

    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.USE_DYRELU = False
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.USE_SE = False

    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.INFERENCE_TH_TRAIN = 0.0
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.PRE_NMS_TOP_N_TRAIN = 3000
    cfg.MODEL.GLIP_MODEL.MODEL.ATSS.POST_NMS_TOP_N_TRAIN = 1000
    # ---------------------------------------------------------------------------- #
    # DYHEAD Options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.NUM_CLASSES = 81  # the number of classes including background
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.PRIOR_PROB = 0.01

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.NUM_CONVS = 4
    # the channels of convolutions used in the cls and bbox tower
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.CHANNELS = 128
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.GROUPS = 1
    # if use deformable conv to align features
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.USE_DFCONV = False

    # topk for selecting candidate positive samples from each level
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.TOPK = 9

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.SCORE_AGG = "MEAN"  # MEAN or MAX, for binary focal loss score aggregation

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.LOG_SCALE = 0.0  # temperature (dot product)
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.SHALLOW_LOG_SCALE = 0.0  # # temperature (shallow contrastive)

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.USE_GN = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.USE_NSYNCBN = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.USE_SYNCBN = False

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.USE_DYFUSE = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.USE_DYRELU = False

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.CONV_FUNC = ''

    # CosineSimOutputLayers: https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/modeling/roi_heads/fast_rcnn.py#L448-L464
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.COSINE_SCALE = -1.0

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.TYPE = ""
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE = 256
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.JOINT_OUT_SIZE = 256
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT = 0.1
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.JOINT_MLP_LAYERS = 2

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS = False

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT = 1.0
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_GAMMA = 2.0
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_ALPHA = 0.25

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM = 64
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_ALIGN_LOSS_WEIGHT = 1.0
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT = 1.0
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE = True
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D = False

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT = False

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT = False

    # Controls for 
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT = False

    # MLM Loss
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES = True
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.NO_MASK_FOR_OD = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.NO_MASK_FOR_GOLD = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_COEF = 1.0
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.MLM_OBJ_FOR_ONLY_POSITIVE  = False

    # Shallow Contrastive Loss (FPN)
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_MAX_POSITIVE_ANCHORS = 100
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_ZERO_PADS = False
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_HIDDEN_DIM = 64
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_LOSS_WEIGHT = 1.0

    # Shallow Contrastive Loss (BACKBONE)
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS = False

    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False

    # use checkpoint to save memory
    cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.USE_CHECKPOINT = False

    # ---------------------------------------------------------------------------- #
    # RPN options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.RPN = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.USE_FPN = False
    # Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
    # Stride of the feature map that RPN is attached.
    # For FPN, number of strides should match number of scales
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.ANCHOR_STRIDE = (16,)
    # RPN anchor aspect ratios
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
    # Anchor shift away ration from the center for r,t,l,d
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.ANCHOR_SHIFT = (0.0, 0.0, 0.0, 0.0)
    # Use center to decide anchor size
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.USE_RELATIVE_SIZE = False
    # Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
    # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.STRADDLE_THRESH = 0
    # Anchor scales per octave for complex anchors
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.OCTAVE = 2.0
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.SCALES_PER_OCTAVE = 3
    # Minimum overlap required between an anchor and ground-truth box for the
    # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
    # ==> positive RPN example)
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
    # Maximum overlap allowed between an anchor and ground-truth box for the
    # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
    # ==> negative RPN example)
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
    # Total number of RPN examples per image
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    # Target fraction of foreground (positive) examples per RPN minibatch
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.POSITIVE_FRACTION = 0.5
    # Number of top scoring RPN proposals to keep before applying NMS
    # When FPN is used, this is *per FPN level* (not total)
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
    # Number of top scoring RPN proposals to keep after applying NMS
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
    # NMS threshold used on RPN proposals
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.NMS_THRESH = 0.7
    # Proposal height and width both need to be greater than RPN_MIN_SIZE
    # (a the scale used during training or inference)
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.MIN_SIZE = 0
    # Number of top scoring RPN proposals to keep after combining proposals from
    # all FPN levels
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
    # Custom rpn head, empty to use default conv or separable conv
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.FREEZE = False
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.FORCE_BOXES = False
    cfg.MODEL.GLIP_MODEL.MODEL.RPN.RETURN_FUSED_FEATURES = False

    # ---------------------------------------------------------------------------- #
    # ROI HEADS options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.USE_FPN = False
    # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
    # Overlap threshold for an RoI to be considered background
    # (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
    # Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
    # These are empirically chosen to approximately lead to unit variance targets
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
    # RoI minibatch size *per image* (number of regions of interest [ROIs])
    # Total number of RoIs per training minibatch =
    #   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
    # E.g., a common configuration is: 512 * 2 * 8 = 8192
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

    # Only used on test mode

    # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
    # balance obtaining high recall with not having too many low precision
    # detections that will slow down inference post processing steps (like NMS)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
    # Overlap threshold used for non-maximum suppression (suppress boxes with
    # IoU >= this threshold)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.NMS = 0.5
    # Maximum number of detections to return per image (100 is based on the limit
    # established for the COCO dataset)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100

    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
    # Hidden layer dimension when using an MLP for the RoI box head
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
    # GN
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.USE_GN = False
    # Dilation
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.DILATION = 1
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4
    # Use D2 style ROIAlignV2
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_BOX_HEAD.POOLER_ALIGNED = False

    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
    # Whether or not resize and translate masks to the input image.
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
    # Dilation
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.DILATION = 1
    # GN
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.USE_GN = False
    # HG
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_MASK_HEAD.HG_SCALE = 1

    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAME = ()  # If left empty, use default names
    cfg.MODEL.GLIP_MODEL.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

    # ---------------------------------------------------------------------------- #
    # ResNe[X]t options (ResNets = {ResNet, ResNeXt}
    # Note that parts of a resnet may be used for both the backbone and the head
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS = CN()

    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.USE_STEM3X3 = False
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.WITH_SE = False
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.USE_AVG_DOWN = False

    # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.NUM_GROUPS = 1

    # Baseline width of each group
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.WIDTH_PER_GROUP = 64

    # Place the stride 2 conv on the 1x1 filter
    # Use True only for the original MSRA ResNet; use False for C2 and Torch models
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.STRIDE_IN_1X1 = True

    # Residual transformation function
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
    # ResNet's stem function (conv1 and pool1)
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

    # Apply dilation in stage "res5"
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.RES5_DILATION = 1

    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.REVISION = "resnet_light"
    # Deformable convolutions
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.WITH_MODULATED_DCN = False
    cfg.MODEL.GLIP_MODEL.MODEL.RESNETS.DEFORMABLE_GROUPS = 1

    # ---------------------------------------------------------------------------- #
    # Swin Transformer
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.OUT_CHANNELS = (96, 192, 384, 768)
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.DEPTHS = (2, 2, 6, 2)
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.NUM_HEADS = (3, 6, 12, 24)
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.APE = False
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.VERSION = "v1"
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.OUT_NORM = True
    cfg.MODEL.GLIP_MODEL.MODEL.SWINT.LAYER_SCALE = 0

    # ---------------------------------------------------------------------------- #
    # CVT SPEC
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.SPEC = CN(new_allowed=True)

    # ---------------------------------------------------------------------------- #
    # CLIP SPEC
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.MODEL.CLIP = CN()
    cfg.MODEL.GLIP_MODEL.MODEL.CLIP.CONTEXT_LENGTH = 256  # default 77
    cfg.MODEL.GLIP_MODEL.MODEL.CLIP.WIDTH = 512
    cfg.MODEL.GLIP_MODEL.MODEL.CLIP.LAYERS = 12
    cfg.MODEL.GLIP_MODEL.MODEL.CLIP.HEADS = 8
    cfg.MODEL.GLIP_MODEL.MODEL.CLIP.DROP_PATH = 0.0
    cfg.MODEL.GLIP_MODEL.MODEL.CLIP.TOKENIZER = "clip"
    cfg.MODEL.GLIP_MODEL.MODEL.CLIP.VOCAB_SIZE = 49408

    # ---------------------------------------------------------------------------- #
    # SEARCH
    # ---------------------------------------------------------------------------- #

    cfg.MODEL.GLIP_MODEL.SEARCH = CN()
    cfg.MODEL.GLIP_MODEL.SEARCH.MAX_EPOCH = 20
    cfg.MODEL.GLIP_MODEL.SEARCH.SELECT_NUM = 20
    cfg.MODEL.GLIP_MODEL.SEARCH.POPULATION_NUM = 64
    cfg.MODEL.GLIP_MODEL.SEARCH.MUTATION_NUM = 24
    cfg.MODEL.GLIP_MODEL.SEARCH.CROSSOVER_NUM = 24
    cfg.MODEL.GLIP_MODEL.SEARCH.MUTATION_PROB = 0.1

    # ---------------------------------------------------------------------------- #
    # Solver
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.SOLVER = CN()
    cfg.MODEL.GLIP_MODEL.SOLVER.USE_AMP = False

    cfg.MODEL.GLIP_MODEL.SOLVER.MAX_ITER = 40000
    cfg.MODEL.GLIP_MODEL.SOLVER.MULTI_MAX_ITER = ()  # set different max epoch for different stage
    cfg.MODEL.GLIP_MODEL.SOLVER.MAX_EPOCH = 0  # any epoch number>0 will overwrite max_iter
    cfg.MODEL.GLIP_MODEL.SOLVER.MULTI_MAX_EPOCH = ()  # set different max epoch for different stage

    cfg.MODEL.GLIP_MODEL.SOLVER.OPTIMIZER = "SGD"  # "ADAMW"

    cfg.MODEL.GLIP_MODEL.SOLVER.BASE_LR = 0.001

    cfg.MODEL.GLIP_MODEL.SOLVER.LANG_LR = 0.00001
    cfg.MODEL.GLIP_MODEL.SOLVER.BACKBONE_BODY_LR_FACTOR = 1.0

    cfg.MODEL.GLIP_MODEL.SOLVER.BIAS_LR_FACTOR = 2
    cfg.MODEL.GLIP_MODEL.SOLVER.GRAD_CLIP = 0.0
    # D2 gradient clip
    cfg.MODEL.GLIP_MODEL.SOLVER.CLIP_GRADIENTS = CN()
    cfg.MODEL.GLIP_MODEL.SOLVER.CLIP_GRADIENTS.ENABLED = False
    cfg.MODEL.GLIP_MODEL.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.0
    cfg.MODEL.GLIP_MODEL.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.MODEL.GLIP_MODEL.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.MODEL.GLIP_MODEL.SOLVER.MODEL_EMA = 0.0

    cfg.MODEL.GLIP_MODEL.SOLVER.MOMENTUM = 0.9

    cfg.MODEL.GLIP_MODEL.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.MODEL.GLIP_MODEL.SOLVER.WEIGHT_DECAY_BIAS = 0.0
    cfg.MODEL.GLIP_MODEL.SOLVER.WEIGHT_DECAY_NORM_FACTOR = 1.0

    # use cosine lr to replace default multistage
    cfg.MODEL.GLIP_MODEL.SOLVER.USE_COSINE = False
    cfg.MODEL.GLIP_MODEL.SOLVER.MIN_LR = 0.000001

    cfg.MODEL.GLIP_MODEL.SOLVER.GAMMA = 0.1
    cfg.MODEL.GLIP_MODEL.SOLVER.STEPS = (30000,)

    cfg.MODEL.GLIP_MODEL.SOLVER.USE_AUTOSTEP = False
    cfg.MODEL.GLIP_MODEL.SOLVER.STEP_PATIENCE = 5

    cfg.MODEL.GLIP_MODEL.SOLVER.WARMUP_FACTOR = 0.3333
    cfg.MODEL.GLIP_MODEL.SOLVER.WARMUP_ITERS = 500
    cfg.MODEL.GLIP_MODEL.SOLVER.WARMUP_METHOD = "linear"

    cfg.MODEL.GLIP_MODEL.SOLVER.CHECKPOINT_PERIOD = 2500
    cfg.MODEL.GLIP_MODEL.SOLVER.CHECKPOINT_PER_EPOCH = -1.0
    cfg.MODEL.GLIP_MODEL.SOLVER.TEST_WITH_INFERENCE = False
    cfg.MODEL.GLIP_MODEL.SOLVER.AUTO_TERMINATE_PATIENCE = -1
    # Number of images per batch
    # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
    # see 2 images per batch
    cfg.MODEL.GLIP_MODEL.SOLVER.IMS_PER_BATCH = 16
    # This is the max negative ratio allowed per batch
    cfg.MODEL.GLIP_MODEL.SOLVER.MAX_NEG_PER_BATCH = 0.1

    cfg.MODEL.GLIP_MODEL.SOLVER.SEED = 0
    cfg.MODEL.GLIP_MODEL.SOLVER.DISABLE_OUTPUT_DISTRIBUTED = False


    cfg.MODEL.GLIP_MODEL.SOLVER.PROMPT_PROBING_LEVEL = -1.0 
    # -1 means tuning the whole model; 
    # 1 means tuning the whole language model; 1.5 means tuning the box head as well

    cfg.MODEL.GLIP_MODEL.SOLVER.FIND_UNUSED_PARAMETERS = True
    cfg.MODEL.GLIP_MODEL.SOLVER.DATASET_LENGTH = -1 # Just for logging purpose
    cfg.MODEL.GLIP_MODEL.SOLVER.TUNING_HIGHLEVEL_OVERRIDE = None
    cfg.MODEL.GLIP_MODEL.SOLVER.USE_EMA_FOR_MONITOR = False

    cfg.MODEL.GLIP_MODEL.SOLVER.WEIGHT_DECAY_SCHEDULE = False
    cfg.MODEL.GLIP_MODEL.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO = 0.667

    # ---------------------------------------------------------------------------- #
    # Specific test options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.TEST = CN()
    cfg.MODEL.GLIP_MODEL.TEST.EXPECTED_RESULTS = []
    cfg.MODEL.GLIP_MODEL.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
    cfg.MODEL.GLIP_MODEL.TEST.DURING_TRAINING = False
    # Number of images per batch
    # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
    # see 2 images per batch
    cfg.MODEL.GLIP_MODEL.TEST.IMS_PER_BATCH = 16
    # Special Test Configuration
    cfg.MODEL.GLIP_MODEL.TEST.USE_MULTISCALE = False
    # cfg.MODEL.GLIP_MODEL.TEST.SCALES = (400, 600, 800, 1000, 1200, 1400)
    # cfg.MODEL.GLIP_MODEL.TEST.RANGES = ((96, 10000), (64, 10000), (0, 10000), (0, 10000), (0, 256), (0, 192))
    cfg.MODEL.GLIP_MODEL.TEST.SCALES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800)
    cfg.MODEL.GLIP_MODEL.TEST.RANGES = ((96, 10000), (96, 10000), (64, 10000), (64, 10000), (64, 10000), (0, 10000), (0, 10000), (0, 256), (0, 256), (0, 192), (0, 192), (0, 96))
    cfg.MODEL.GLIP_MODEL.TEST.MAX_SIZE = 2500
    cfg.MODEL.GLIP_MODEL.TEST.FLIP = True
    cfg.MODEL.GLIP_MODEL.TEST.SPECIAL_NMS = 'none'  # ('none', 'soft-nms', 'vote', 'soft-vote')
    cfg.MODEL.GLIP_MODEL.TEST.TH = 0.6  # threshold for nms or vote
    cfg.MODEL.GLIP_MODEL.TEST.PRE_NMS_TOP_N = 1000
    cfg.MODEL.GLIP_MODEL.TEST.NUM_CLASSES = 81
    cfg.MODEL.GLIP_MODEL.TEST.SELECT_CLASSES = ()

    cfg.MODEL.GLIP_MODEL.TEST.EVAL_TASK = ""
    cfg.MODEL.GLIP_MODEL.TEST.SUBSET = -1
    cfg.MODEL.GLIP_MODEL.TEST.CHUNKED_EVALUATION = -1
    cfg.MODEL.GLIP_MODEL.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM = -1
    # ---------------------------------------------------------------------------- #
    # Misc options
    # ---------------------------------------------------------------------------- #
    cfg.MODEL.GLIP_MODEL.OUTPUT_DIR = "OUTPUT"

    cfg.MODEL.GLIP_MODEL.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

    # TensorBoard experiment location
    cfg.MODEL.GLIP_MODEL.TENSORBOARD_EXP = "OUTPUT"


    cfg.MODEL.GLIP_MODEL.GLIPKNOW = CN()
    cfg.MODEL.GLIP_MODEL.GLIPKNOW.KNOWLEDGE_FILE = ""
    cfg.MODEL.GLIP_MODEL.GLIPKNOW.KNOWLEDGE_TYPE = ""
    cfg.MODEL.GLIP_MODEL.GLIPKNOW.MAX_NUM_CLASSES_PER_BATCH_TRAIN = -1
    cfg.MODEL.GLIP_MODEL.GLIPKNOW.PARALLEL_LANGUAGE_INPUT = False
    cfg.MODEL.GLIP_MODEL.GLIPKNOW.LAN_FEATURE_AGG_TYPE = "first"
    cfg.MODEL.GLIP_MODEL.GLIPKNOW.GPT3_NUM = 5
    cfg.MODEL.GLIP_MODEL.GLIPKNOW.WIKI_AND_GPT3 = False