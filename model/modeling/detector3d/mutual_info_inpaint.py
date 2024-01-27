import pdb
import cv2
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

from mmcv.runner.base_module import BaseModule

import torch
from torch import nn
import torch.nn.functional as F

class MutualInfoInpainter(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.inpaint_loss_weight = 100
        self.pts_paint_cam = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.PTS_PAINT_CAM
        self.cam_paint_pts = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CAM_PAINT_PTS


    def forward(self, cam_bev_feat, point_bev_feat, update_cam_bev_feat, update_point_bev_feat, cam_valid_mask, pts_valid_mask):
        '''
        Input:
            cam_bev_feat shape: (B, C, bev_y, bev_x)
            point_bev_feat: (B, C, bev_y, bev_x)
            update_cam_bev_feat: (B, C, bev_y, bev_x)
            update_point_bev_feat: (B, C, bev_y, bev_x)
            cam_valid_mask: (B, bev_y, bev_x)
            pts_valid_mask: (B, bev_y, bev_x)
        '''
        inpaint_loss = 0
        # Supervise camera based on point (The valid region of point cloud BEV)
        if self.pts_paint_cam:
            cam_inpaint_loss = self.inpaint_loss_weight * F.l1_loss(update_cam_bev_feat * pts_valid_mask.unsqueeze(1), (point_bev_feat * pts_valid_mask.unsqueeze(1)).detach(), reduction = 'mean')
            inpaint_loss = inpaint_loss + cam_inpaint_loss
        # Supervise point based on camera (The region valid for camera BEV while invalid for point BEV)
        if self.cam_paint_pts:
            supervise_point_mask = cam_valid_mask & (~pts_valid_mask)
            point_inpaint_loss = self.inpaint_loss_weight * F.l1_loss(update_point_bev_feat * supervise_point_mask.unsqueeze(1), (cam_bev_feat * supervise_point_mask.unsqueeze(1)).detach(), reduction = 'mean')
            inpaint_loss = inpaint_loss + point_inpaint_loss
        
        return inpaint_loss
