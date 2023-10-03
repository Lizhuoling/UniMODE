import pdb
import cv2
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from mmdet.models.backbones.resnet import BasicBlock

class MultiModal_DualView_Attn(nn.Module):
    def __init__(self, cfg, img_in_ch, point_in_ch, voxel_num, out_ch):
        super(MultiModal_DualView_Attn, self).__init__()
        self.cfg = cfg
        self.img_in_ch = img_in_ch
        self.point_in_ch = point_in_ch
        self.voxel_num = voxel_num  # (voxel_x, voxel_y, voxel_z)
        self.out_ch = out_ch
        
        self.occ_mlp1 = nn.Linear(point_in_ch, self.voxel_num[1])
        self.mm_fuse1 = nn.Sequential(
            nn.Conv2d(self.img_in_ch + self.point_in_ch, self.point_in_ch, 1, bias=True),
            BasicBlock(self.point_in_ch, self.point_in_ch),
        )

        self.occ_mlp2 = nn.Linear(point_in_ch, self.voxel_num[1])
        self.mm_fuse2 = nn.Sequential(
            nn.Conv2d(self.img_in_ch + self.point_in_ch, self.out_ch, 1, bias=True),
            BasicBlock(self.out_ch, self.out_ch),
        )

    def forward(self, cam_voxel_feat, point_feat, batched_inputs):
        '''
        Input:
            cam_voxel_feat shape: (B, cam_C, voxel_z, voxel_y, voxel_x)
            point_feat shape: (B, point_C, voxel_z, voxel_x)
        '''
        B, _, voxel_z, voxel_y, voxel_x = cam_voxel_feat.shape

        # vis
        '''cam_feat = cam_voxel_feat.sum(3).sum(1)[0]
        cam_feat = cam_feat.detach().cpu().numpy()
        pdb.set_trace()
        plt.imshow(cam_feat)
        plt.savefig('vis.png')'''

        occ_pred1 = self.occ_mlp1(point_feat.view(B, self.point_in_ch, voxel_z * voxel_x).permute(0, 2, 1)).sigmoid()   # Left shape: (B, voxel_z*voxel_x, voxel_y)
        occ_pred1 = occ_pred1.view(B, voxel_z, voxel_x, voxel_y).permute(0, 1, 3, 2).unsqueeze(1)   # Left shape: (B, 1, voxel_z, voxel_y, voxel_x)
        cam_bev_feat1 = (cam_voxel_feat * occ_pred1).sum(dim = 3)   # Left shape: (B, cam_C, voxel_z, voxel_x)

        point_feat2 = torch.cat((cam_bev_feat1, point_feat), dim = 1)
        point_feat2 = self.mm_fuse1(point_feat2)    # Left shape: (B, point_C, voxel_z, voxel_x)

        occ_pred2 = self.occ_mlp2(point_feat2.view(B, self.point_in_ch, voxel_z * voxel_x).permute(0, 2, 1)).sigmoid()   # Left shape: (B, voxel_z*voxel_x, voxel_y)
        occ_pred2 = occ_pred2.view(B, voxel_z, voxel_x, voxel_y).permute(0, 1, 3, 2).unsqueeze(1)   # Left shape: (B, 1, voxel_z, voxel_y, voxel_x)
        cam_bev_feat2 = (cam_voxel_feat * occ_pred2).sum(dim = 3)   # Left shape: (B, cam_C, voxel_z, voxel_x)

        bev_feat = torch.cat((cam_bev_feat2, point_feat2), dim = 1)
        bev_feat = self.mm_fuse2(bev_feat)    # Left shape: (B, out_C, voxel_z, voxel_x)
        
        return bev_feat