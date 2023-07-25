import pdb
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from mmdet.core import multi_apply

from cubercnn.util import torch_dist


class CENTER_HEAD(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.proposal_number = cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.PROPOSAL_NUMBER
        self.downsample_factor = self.cfg.MODEL.DETECTOR3D.PETR.DOWNSAMPLE_FACTOR
        self.embed_dims = 256
        self.obj_loss_weight = cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.OBJ_LOSS_WEIGHT
        self.depth_loss_weight = cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.DEPTH_LOSS_WEIGHT
        self.offset_loss_weight = cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.OFFSET_LOSS_WEIGHT

        # Predict the confidence that concerned objects exist. It works like RPN.
        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, self.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dims, momentum=0.1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=1 // 2, bias=True),
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, self.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dims, momentum=0.1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=0, bias=True)
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, self.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dims, momentum=0.1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, 2, kernel_size=1, padding=0, bias=True)
        )

        self.cls_loss_fnc = FocalLoss(2, 4)

        self.init_weights()

    def init_weights(self):
        INIT_P = 0.01
        self.obj_head[-1].bias.data.fill_(- np.log(1 / INIT_P - 1))
    
    def forward(self, feat, Ks):
        obj_pred = sigmoid_hm(self.obj_head(feat))  # Left shape: (B, 1, feat_h, feat_w)
        depth_pred = self.depth_head(feat)  # Left shape: (B, 1, feat_h, feat_w)
        offset_pred = self.offset_head(feat)    # Left shape: (B, 2, feat_h, feat_w)

        bs, _, feat_h, feat_w = obj_pred.shape
        center_conf, center_idx = obj_pred.view(bs, feat_h * feat_w).topk(self.proposal_number, dim = -1)
        center_u, center_v = center_idx % feat_w, center_idx // feat_h
        center_uv = torch.stack((center_u, center_v), dim = -1).float()  # Left shape: (B, num_proposal, 2)
        
        norm_center_u = (center_u - feat_w / 2) / (feat_w / 2)  # Left shape: (B, num_proposal)
        norm_center_v = (center_v - center_v / 2) / (feat_h / 2)
        norm_center_uv = torch.stack((norm_center_u, norm_center_v), dim = -1)  # Left shape: (B, num_proposal, 2)
        
        topk_depth_pred = F.grid_sample(depth_pred, norm_center_uv.unsqueeze(2)).squeeze(-1).permute(0, 2, 1).contiguous()   # Left shape: (B, num_proposal, 1)
        topk_offset_pred = F.grid_sample(offset_pred, norm_center_uv.unsqueeze(2)).squeeze(-1).permute(0, 2, 1).contiguous()   # Left shape: (B, num_proposal, 2)
        center_uv = (center_uv + topk_offset_pred) * self.downsample_factor # Left shape: (B, num_proposal, 2)
        center_uvd = torch.cat((center_uv * topk_depth_pred, topk_depth_pred), dim = -1)    # Left shape: (B, num_proposal, 3)
        
        center_conf = center_conf.unsqueeze(-1) # Left shape: (B, num_proposal, 1)
        center_xyz = (Ks.unsqueeze(1).inverse() @ center_uvd.unsqueeze(-1)).squeeze(-1) # Left shape: (B, num_proposal, 3)

        return dict(
            obj_pred = obj_pred,
            depth_pred = depth_pred,
            offset_pred = offset_pred,
            topk_center_conf = center_conf,
            topk_center_xyz = center_xyz,
        )

    def loss(self, center_head_out, Ks, batched_inputs):
        obj_preds = center_head_out['obj_pred']  # Left shape: (B, 1, feat_h, feat_w)
        depth_preds = center_head_out['depth_pred']   # Left shape: (B, 1, feat_h, feat_w)
        offset_preds = center_head_out['offset_pred']   # Left shape: (B, 2, feat_h, feat_w)

        loss_info_all_batch = multi_apply(self.loss_single, obj_preds, depth_preds, offset_preds, batched_inputs)[0]
        
        loss_dict = {}
        total_num_gt = sum([ele['num_gt'] for ele in loss_info_all_batch])
        total_num_gt = torch_dist.reduce_mean(torch.Tensor([total_num_gt]).cuda()).item()
        torch_dist.synchronize()

        for key in loss_info_all_batch[0].keys():
            if key != 'num_gt':
                loss_dict[key] = sum([ele[key] for ele in loss_info_all_batch]) / total_num_gt
        return loss_dict

    def loss_single(self, obj_pred, depth_pred, offset_pred, batch_input):
        '''
        Input:
            obj_pred shape: (1, feat_h, feat_w)
            depth_pred shape: (1, feat_h, feat_w)
            offset_pred shape: (2, feat_h, feat_w)
        '''
        device = obj_pred.device
        _, feat_h, feat_w = obj_pred.shape

        # Objectness loss
        gt_heatmaps = batch_input['obj_heatmap'][None][None].to(device) # Left shape: (1, 1, feat_h, feat_w)
        obj_loss, num_obj_pos = self.cls_loss_fnc(obj_pred[None], gt_heatmaps)
        obj_loss = self.obj_loss_weight * obj_loss / torch.clamp(num_obj_pos, 1)

        # Regression
        instances = batch_input['instances']
        gt_box2d_center = instances.get('gt_featreso_box2d_center').to(device) # Left shape: (num_box, 2)
        gt_2dto3d_offset = instances.get('gt_featreso_2dto3d_offset').to(device)    # Left shape: (num_box, 2)
        gt_depth = instances.get('gt_boxes3D')[:, 2:3].to(device)  # Left shape: (num_box, 1)
        num_gt = gt_depth.shape[0]
        
        gt_box2d_center[:, 0] = (gt_box2d_center[:, 0] - feat_w / 2) / (feat_w / 2)
        gt_box2d_center[:, 1] = (gt_box2d_center[:, 1] - feat_h / 2) / (feat_h / 2)
        depth_pred = F.grid_sample(depth_pred[None], gt_box2d_center[None, None])[0, :, 0, :].permute(1, 0).contiguous() # Left shape: (num_box, 1)
        depth_pred = 1 / torch.sigmoid(depth_pred) - 1   # inv_sigmoid decoding
        offset_pred = F.grid_sample(offset_pred[None], gt_box2d_center[None, None])[0, :, 0, :].permute(1, 0).contiguous()  # Left shape: (num_box, 2)
        
        depth_loss = self.depth_loss_weight * F.l1_loss(depth_pred, gt_depth, reduction = 'sum')
        offset_loss = self.offset_loss_weight * F.l1_loss(offset_pred, gt_2dto3d_offset, reduction = 'sum')

        return (dict(
            center_obj_loss = obj_loss,
            center_depth_loss = depth_loss,
            center_offset_loss = offset_loss,
            num_gt = num_gt,
        ),)

        

def sigmoid_hm(hm_features):
    x = hm_features.sigmoid_()
    x = x.clamp(min=1e-4, max=1 - 1e-4)
    return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = (target.lt(1) & target.ge(0)).float()
        ignore_index = target.eq(-1).float() # ignored pixels

        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = torch.log(prediction) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
                        
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        loss = - negative_loss - positive_loss

        return loss, num_positive