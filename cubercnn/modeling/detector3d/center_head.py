import pdb
import cv2
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from mmcv.cnn import build_conv_layer
from mmdet.core import multi_apply
from mmdet.models.backbones.resnet import BasicBlock

from cubercnn.util import torch_dist
from cubercnn.modeling.detector3d.depthnet import ASPP


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

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.embed_dims, kernel_size=1, padding=1 // 2, bias=True),
            BasicBlock(self.embed_dims, self.embed_dims),
        )

        self.obj_head = nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=1 // 2, bias=True)

        self.depth_head = nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=0, bias=True)

        self.offset_head = nn.Conv2d(self.embed_dims, 2, kernel_size=1, padding=0, bias=True)

        self.cls_loss_fnc = FocalLoss(2, 4)

        self.init_weights()

    def init_weights(self):
        INIT_P = 0.01
        self.obj_head.bias.data.fill_(- np.log(1 / INIT_P - 1))
    
    def forward(self, feat, Ks, batched_inputs):
        feat = self.shared_conv(feat)
        
        obj_pred = sigmoid_hm(self.obj_head(feat))  # Left shape: (B, 1, feat_h, feat_w)
        depth_pred = self.depth_head(feat)  # Left shape: (B, 1, feat_h, feat_w)
        depth_pred = 1 / depth_pred.sigmoid() - 1   # inv_sigmoid decoding
        if self.cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.FOCAL_DECOUPLE:
            virtual_focal_y = self.cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.VIRTUAL_FOCAL_Y
            real_focal_y = Ks[:, 1, 1]
            depth_pred = depth_pred * real_focal_y[:, None, None, None] / virtual_focal_y
        offset_pred = self.offset_head(feat)    # Left shape: (B, 2, feat_h, feat_w)

        bs, _, feat_h, feat_w = obj_pred.shape

        # Predicted 3D points, which are used for initializing queries during training and also determining BEV perception range during inference.
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

        tgt_depth_preds = []
        tgt_offset_preds = []
        tgt_conf_preds = []
        tgt_xyz_preds = []
        if self.training:
            # During training, we get proposal points by selecting gt centers.
            for bs_idx, batch_input in enumerate(batched_inputs):
                gt_box2d_center = batch_input['instances'].get('gt_featreso_box2d_center').cuda() # Left shape: (num_box, 2)
                norm_gt_box2d_center = gt_box2d_center.clone()
                norm_gt_box2d_center[:, 0] = (norm_gt_box2d_center[:, 0] - feat_w / 2) / (feat_w / 2)
                norm_gt_box2d_center[:, 1] = (norm_gt_box2d_center[:, 1] - feat_h / 2) / (feat_h / 2)
                # Depth prediction
                tgt_depth_pred = F.grid_sample(depth_pred[bs_idx][None], norm_gt_box2d_center[None, None])[0, :, 0, :].permute(1, 0).contiguous()    # Left shape: (num_gt, 1)
                tgt_depth_preds.append(tgt_depth_pred)
                # Offset prediction
                tgt_offset_pred = F.grid_sample(offset_pred[bs_idx][None], norm_gt_box2d_center[None, None])[0, :, 0, :].permute(1, 0).contiguous()  # Left shape: (num_gt, 2)
                tgt_offset_preds.append(tgt_offset_pred)
                # Conf prediction
                gt_conf_pred = F.grid_sample(obj_pred[bs_idx][None], norm_gt_box2d_center[None, None])[0, :, 0, :].permute(1, 0).contiguous()  # Left shape: (num_gt, 1)
                tgt_conf_preds.append(gt_conf_pred)

                tgt_3dcenter_uv_pred = (gt_box2d_center + tgt_offset_pred) * self.downsample_factor   # Left shape: (num_gt, 2)
                tgt_3dcenter_uvd_pred = torch.cat((tgt_3dcenter_uv_pred * tgt_depth_pred, tgt_depth_pred), dim = -1)    # Left shape: (num_gt, 3)
                tgt_3dcenter_xyz_pred = (Ks[bs_idx][None].inverse() @ tgt_3dcenter_uvd_pred.unsqueeze(-1)).squeeze(-1)  # Left shape: (num_gt, 3)
                tgt_xyz_preds.append(tgt_3dcenter_xyz_pred)
        else:
            # During inference, we obtain proposal points by confidence filtering
            valid_proposal_mask = (center_conf > self.cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.PROPOSAL_CONF_THRE).squeeze(-1) # Left shape: (B, valid_proposal_num)
            for bs_idx, bs_mask in enumerate(valid_proposal_mask):
                bs_valid_center_conf = center_conf[bs_idx][bs_mask]
                tgt_conf_preds.append(bs_valid_center_conf) # Left shape: (num_valid_proposal, 1)
                bs_valid_center_xyz = center_xyz[bs_idx][bs_mask] # Left shape: (num_valid_proposal, 3)
                tgt_xyz_preds.append(bs_valid_center_xyz)


        return dict(
            obj_pred = obj_pred,    # Used for computing loss.
            topk_center_conf = center_conf.detach(),    # Used for initializing queries.
            topk_center_xyz = center_xyz.detach(),  # Used for initializing queries.
            tgt_conf_preds = tgt_conf_preds,
            tgt_xyz_preds = tgt_xyz_preds, 
            tgt_depth_preds = tgt_depth_preds,  # Used for computing loss.
            tgt_offset_preds = tgt_offset_preds,    # Used for computing loss.
        )

    def loss(self, center_head_out, Ks, batched_inputs):
        obj_preds = center_head_out['obj_pred']  # Left shape: (B, 1, feat_h, feat_w)
        tgt_depth_preds = center_head_out['tgt_depth_preds']   # A list and the length is bs
        tgt_offset_preds = center_head_out['tgt_offset_preds']   # A list and the length is bs

        loss_info_all_batch = multi_apply(self.loss_single, obj_preds, tgt_depth_preds, tgt_offset_preds, batched_inputs)[0]
        
        loss_dict = {}
        total_num_gt = sum([ele['num_gt'] for ele in loss_info_all_batch])
        total_num_gt = torch_dist.reduce_mean(torch.Tensor([total_num_gt]).cuda()).item()
        torch_dist.synchronize()

        for key in loss_info_all_batch[0].keys():
            if key != 'num_gt':
                loss_dict[key] = sum([ele[key] for ele in loss_info_all_batch]) / total_num_gt
        return loss_dict

    def loss_single(self, obj_pred, tgt_depth_pred, tgt_offset_pred, batch_input):
        '''
        Input:
            obj_pred shape: (1, feat_h, feat_w)
            tgt_depth_pred shape: (num_gt, 1)
            offset_pred shape: (num_gt, 2)
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

        depth_pred = tgt_depth_pred # Left shape: (num_box, 1)
        offset_pred = tgt_offset_pred  # Left shape: (num_box, 2)

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