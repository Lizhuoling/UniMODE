# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import pdb
import torch
import cv2
import numpy as np
import logging
from typing import List, Optional, Union
from detectron2.structures import BoxMode, Keypoints
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data import (
    DatasetMapper
)
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
)
from detectron2.config import configurable

from .augmentation import PhotoMetricDistortionMultiViewImage

class DatasetMapper3D(DatasetMapper):
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        cfg,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        self.cfg = cfg
        if self.cfg.INPUT.COLOR_AUG and self.is_train:
            self.color_aug = PhotoMetricDistortionMultiViewImage()
        else:
            self.color_aug = None

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = detection_utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "cfg": cfg,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        detection_utils.check_image_size(dataset_dict, image)   # Check whether the image shape match the height and width specified in dataset_dict

        if self.color_aug != None:
            image = self.color_aug(image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # no need for additoinal processing at inference
        if not self.is_train:
            return dataset_dict

        if "annotations" in dataset_dict:
            dataset_id = dataset_dict['dataset_id']
            K = np.array(dataset_dict['K'])

            unknown_categories = self.dataset_id_to_unknown_cats[dataset_id]

            # transform and pop off annotations
            annos = [
                transform_instance_annotations(obj, transforms, K=K)
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]
        
            # convert to instance format
            instances = annotations_to_instances(annos, image_shape, unknown_categories)
            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        dataset_dict["horizontal_flip_flag"] = False
        for transform in transforms:
            if isinstance(transform, T.HFlipTransform):
                dataset_dict["horizontal_flip_flag"] = True

        return dataset_dict

'''
Cached for mirroring annotations
'''
_M1 = np.array([
    [1, 0, 0], 
    [0, -1, 0],
    [0, 0, -1]
])
_M2 = np.array([
    [-1.,  0.,  0.],
    [ 0., -1.,  0.],
    [ 0.,  0.,  1.]
])


def transform_instance_annotations(annotation, transforms, *, K):
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox = transforms.apply_box(np.array([bbox]))[0]
    
    annotation["bbox"] = bbox
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if annotation['center_cam'][2] != 0:

        # project the 3D box annotation XYZ_3D to screen 
        point3D = annotation['center_cam']
        point2D = K @ np.array(point3D)
        point2D[:2] = point2D[:2] / point2D[-1]
        annotation["center_cam_proj"] = point2D.tolist()

        # apply coords transforms to 2D box
        annotation["center_cam_proj"][0:2] = transforms.apply_coords(
            point2D[np.newaxis][:, :2]
        )[0].tolist()

        keypoints = (K @ np.array(annotation["bbox3D_cam"]).T).T
        keypoints[:, 0] /= keypoints[:, -1]
        keypoints[:, 1] /= keypoints[:, -1]
        
        if annotation['ignore']:
            # all keypoints marked as not visible 
            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 1
        else:
            
            valid_keypoints = keypoints[:, 2] > 0

            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 2
            keypoints[valid_keypoints, 2] = 2

        # in place
        transforms.apply_coords(keypoints[:, :2])
        annotation["keypoints"] = keypoints.tolist()

        # manually apply mirror for pose
        for transform in transforms:
            
            # horrizontal flip?
            if isinstance(transform, T.HFlipTransform):

                pose = _M1 @ np.array(annotation["pose"]) @ _M2
                annotation["pose"] = pose.tolist()
                annotation["R_cam"] = pose.tolist()

    return annotation


def annotations_to_instances(annos, image_size, unknown_categories):

    # init
    target = Instances(image_size)
    
    # add classes, 2D boxes, 3D boxes and poses
    target.gt_classes = torch.tensor([int(obj["category_id"]) for obj in annos], dtype=torch.int64)
    target.gt_boxes = Boxes([BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos])
    target.gt_boxes3D = torch.FloatTensor([anno['center_cam_proj'] + anno['dimensions'] + anno['center_cam'] for anno in annos])
    target.gt_poses = torch.FloatTensor([anno['pose'] for anno in annos])
    
    n = len(target.gt_classes)

    # do keypoints?
    target.gt_keypoints = Keypoints(torch.FloatTensor([anno['keypoints'] for anno in annos]))

    gt_unknown_category_mask = torch.zeros(max(unknown_categories)+1, dtype=bool)
    gt_unknown_category_mask[torch.tensor(list(unknown_categories))] = True

    # include available category indices as tensor with GTs
    target.gt_unknown_category_mask = gt_unknown_category_mask.unsqueeze(0).repeat([n, 1])

    return target
