# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, List, Optional
import pdb
import numpy as np
import cv2
import copy
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.config import CfgNode as CN
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.data import MetadataCatalog

from model.modeling.detector3d.build import build_3d_detector

@META_ARCH_REGISTRY.register()
class MM_UniMODE(BaseModule):
    def __init__(self, cfg,):
        super().__init__()
        self.cfg = cfg
        self.fp16_enabled = True
        self.img_mean = torch.Tensor(self.cfg.MODEL.PIXEL_MEAN)[:, None, None]
        self.img_std = torch.Tensor(self.cfg.MODEL.PIXEL_STD)[:, None, None]

        ### Build the 3D Det model ###
        self.detector = build_3d_detector(cfg)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)
        points = [batch_input['point_cloud'].cuda() for batch_input in batched_inputs]
        ori_img_resolution = torch.Tensor([(img.shape[2], img.shape[1]) for img in images]).to(images.device)   # Left shape: (bs, 2)
        
        if self.training:
            batched_inputs = self.remove_invalid_gts(batched_inputs)
        
        # 3D Det
        out = self.detector(images, points, batched_inputs)
        return out

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].cuda() for x in batched_inputs]
        images = [(x - self.img_mean.to(x.device)) / self.img_std.to(x.device) for x in images]

        images = ImageList.from_tensors(images)
        return images
    
    def remove_invalid_gts(self, batched_inputs):
        '''
            Description: Remove the labels the class IDs of which are -1.
        '''
        for batch_input in batched_inputs:
            cls_gts = batch_input['instances'].get('gt_classes')
            valid_gt_mask = (cls_gts != -1)
            for gt_key in ['gt_classes', 'gt_boxes', 'gt_boxes3D', 'gt_poses', 'gt_keypoints', 'gt_unknown_category_mask', 'gt_featreso_box2d_center', 'gt_featreso_2dto3d_offset']:
                if gt_key in ('gt_boxes', 'gt_keypoints'):
                    batch_input['instances']._fields[gt_key].tensor = batch_input['instances']._fields[gt_key].tensor[valid_gt_mask]
                else:
                    if gt_key in batch_input['instances']._fields.keys():
                        batch_input['instances']._fields[gt_key] = batch_input['instances']._fields[gt_key][valid_gt_mask]

        return batched_inputs

def vis_2d_det(boxlist, img, class_names, conf_thre = 0.5):
    img = np.ascontiguousarray(img)
    box_locs = boxlist.bbox.cpu().numpy().astype(np.int32)
    box_labels = boxlist.extra_fields['labels'].cpu().numpy()
    box_confs = boxlist.extra_fields['scores'].cpu().numpy()

    valid_mask = (box_confs > conf_thre) & (box_labels < len(class_names))
    box_locs = box_locs[valid_mask]
    box_labels = box_labels[valid_mask]
    box_confs = box_confs[valid_mask]

    for cnt, box_loc in  enumerate(box_locs):
        box_label = box_labels[cnt]
        class_name = class_names[box_label]
        img = cv2.rectangle(img, box_loc[0:2], box_loc[2:4], (0, 255, 0))
        img = cv2.putText(img, "{:s}: {:.2f}".format(class_name, box_confs[cnt]), box_loc[0:2], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    cv2.imwrite('vis.png', img)
    pdb.set_trace()
