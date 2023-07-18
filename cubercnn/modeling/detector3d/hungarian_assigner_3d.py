# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import pdb
import math
import numpy  as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
#from mmdet.models.task_modules import BBOX_ASSIGNERS, build_match_cost
#from mmdet.models.layers.transformer import inverse_sigmoid

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.match_costs import build_match_cost

from scipy.optimize import linear_sum_assignment
from pytorch3d.transforms import rotation_6d_to_matrix
from pytorch3d.transforms.so3 import so3_relative_angle
from cubercnn.util.util import box_cxcywh_to_xyxy, generalized_box_iou

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self, cls_weight=1.0, reg_weight = 1.0, det2d_l1_weight = 5.0, det2d_iou_weight = 2.0, total_cls_num = None, uncern_range = (-10, 10), cfg = None):
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.det2d_l1_weight = det2d_l1_weight
        self.det2d_iou_weight = det2d_iou_weight
        self.total_cls_num = total_cls_num
        self.uncern_range = uncern_range
        self.cfg = cfg
        #self.cls_loss = build_match_cost({'type': 'FocalLossCost', 'weight': 1.0})
        #self.reg_cost = build_match_cost({'type': 'BBox3DL1Cost', 'weight': 1.0})
        self.reg_loss = nn.L1Loss(reduction = 'none')
        self.iou_cost = build_match_cost({'type': 'IoUCost', 'weight': 1.0})

    @torch.no_grad()
    def assign(self, bbox_preds, cls_scores, batch_input, reg_key_manager, ori_img_resolution, h_scale_ratio, Ks, eps=1e-7):
        """
        Input:
            bbox_preds shape: (num_query, attr_num), attr order: ('loc', 'dim', 'pose', 'uncern')
            cls_scores shape: (num_query, cls_num)
            batch_input: Input dict of a batch. Element keys in batched_input['instances']: ('gt_classes', 'gt_boxes', 'gt_boxes3D', 'gt_poses', 'gt_keypoints', 'gt_unknown_category_mask')
        """
        gt_instance = batch_input['instances']
        num_preds, num_gts = bbox_preds.size(0), gt_instance.get('gt_classes').size(0)
        device = bbox_preds.device

        loc_preds = bbox_preds[..., reg_key_manager('loc')] # Left shape: (num_query, 3)
        dim_preds = bbox_preds[..., reg_key_manager('dim')] # Left shape: (num_query, 3)
        pose_preds = bbox_preds[..., reg_key_manager('pose')] # Left shape: (num_query, 6)
        pose_preds = rotation_6d_to_matrix(pose_preds)  # Left shape: (num_query, 3, 3)
        uncern_preds = bbox_preds[..., reg_key_manager('uncern')] # Left shape: (num_query, 1)
        uncern_preds = torch.clamp(uncern_preds, min = self.uncern_range[0],  max = self.uncern_range[1])
        
        cls_gts = gt_instance.get('gt_classes').to(device) # Left shape: (num_gt,)
        loc_gts = gt_instance.get('gt_boxes3D')[..., 6:9].to(device) # Left shape: (num_gt, 3)
        dim_gts = gt_instance.get('gt_boxes3D')[..., 3:6].to(device) # Left shape: (num_gt, 3)
        pose_gts = gt_instance.get('gt_poses').to(device) # Left shape: (num_gt, 3, 3)
            
        if self.cfg.MODEL.DETECTOR3D.PETR.HEAD.PERFORM_2D_DET:
            det2d_xywh_preds = bbox_preds[..., reg_key_manager('det2d_xywh')] # Left shape: (num_query, 4)
            det2d_xywh_gts = gt_instance.get('gt_boxes').to(device).tensor  # Left shape: (num_gt, 4)
            img_scale = torch.cat((ori_img_resolution, ori_img_resolution), dim = 0)    # Left shape: (4,)
            det2d_xywh_gts = det2d_xywh_gts / img_scale[None]   # Left shape: (num_gt, 4)

        indices = []
        if num_gts == 0:
            indices.append([[], []])
            return [(torch.as_tensor(i, dtype = torch.int64), torch.as_tensor(j, dtype = torch.int64)) for i, j in indices]

        expand_cls_scores = cls_scores.unsqueeze(1).expand(-1, num_gts, -1)    # Left shape: (num_query, num_gt, num_cls)
        expand_cls_gts = F.one_hot(cls_gts, num_classes = self.total_cls_num)[None].expand(num_preds, -1, -1)   # Left shape: (num_query, num_gt, num_cls)
        cls_cost = self.cls_weight * torchvision.ops.sigmoid_focal_loss(expand_cls_scores, expand_cls_gts.float(), reduction = 'none').sum(-1)
        
        loc_preds = loc_preds.unsqueeze(1).expand(-1, num_gts, -1)  # Left shape: (num_query, num_gt, 3)
        uncern_preds = uncern_preds.unsqueeze(1).expand(-1, num_gts, -1)  # Left shape: (num_query, num_gt, 1)
        loc_gts = loc_gts[None].expand(num_preds, -1, -1)   # Left shape: (num_query, num_gt, 3)
        loc_cost = self.reg_weight * (math.sqrt(2) * self.reg_loss(loc_preds, loc_gts).sum(-1, keepdim=True) / uncern_preds.exp() + uncern_preds).squeeze(-1) # Left shape: (num_query, num_gt)
            
        dim_preds = dim_preds.unsqueeze(1).expand(-1, num_gts, -1)  # Left shape: (num_query, num_gt, 3)
        dim_gts = dim_gts[None].expand(num_preds, -1, -1).log()   # Left shape: (num_query, num_gt, 1)
        dim_cost = self.reg_weight * self.reg_loss(dim_preds, dim_gts).sum(-1)  # Left shape: (num_query, num_gt)

        pose_preds = pose_preds.unsqueeze(1).expand(-1, num_gts, -1, -1).reshape(num_preds * num_gts, 3, 3)    # Left shape: (num_query, num_gt, 3, 3)
        pose_gts = pose_gts[None].expand(num_preds, -1, -1, -1).reshape(num_preds * num_gts, 3, 3) # Left shape: (num_query, num_gt, 3, 3)
        cost_pose = self.reg_weight * (1-so3_relative_angle(pose_preds, pose_gts, eps=1, cos_angle=True)).view(num_preds, num_gts)  # Left shape: (num_query, num_gt)

        if self.cfg.MODEL.DETECTOR3D.PETR.HEAD.PERFORM_2D_DET:
            det2d_reg_loss = self.det2d_l1_weight * self.reg_loss(det2d_xywh_preds[:, None].expand(-1, num_gts, -1), 
                det2d_xywh_gts[None].expand(num_preds, -1, -1)).sum(-1)  # Left shape: (num_query, num_gt)
            det2d_iou_loss = -self.det2d_iou_weight * generalized_box_iou(
                box_cxcywh_to_xyxy(det2d_xywh_preds),
                box_cxcywh_to_xyxy(det2d_xywh_gts)
            )   # Left shape: (num_query, num_gt)
        else:
            det2d_reg_loss = 0
            det2d_iou_loss = 0

        cost = cls_cost + loc_cost + dim_cost + cost_pose + det2d_reg_loss + det2d_iou_loss # Left shape: (num_query, num_gt)

        indices.append(linear_sum_assignment(cost.cpu().numpy()))
        
        return [(torch.as_tensor(i, dtype = torch.int64), torch.as_tensor(j, dtype = torch.int64)) for i, j in indices]