import pdb
import cv2
import copy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models import build_backbone
from mmdet.models import build_neck
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import build_positional_encoding

from cubercnn.util.grid_mask import GridMask
from cubercnn.modeling import neck

class DETECTOR_PETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.img_mean = torch.Tensor(self.cfg.MODEL.PIXEL_MEAN)  # (b, g, r)
        self.img_std = torch.Tensor(self.cfg.MODEL.PIXEL_STD)

        if self.cfg.MODEL.DETECTOR3D.PETR.GRID_MASK:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        else:
            self.grid_mask = False

        backbone_cfg = backbone_cfgs(cfg.MODEL.DETECTOR3D.PETR.BACKBONE_NAME)
        self.img_backbone = build_backbone(backbone_cfg)
        self.img_backbone.init_weights()

        neck_cfg = neck_cfgs(cfg.MODEL.DETECTOR3D.PETR.NECK_NAME)
        self.img_neck = build_neck(neck_cfg)

        self.petr_head = PETR_HEAD(cfg)

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, imgs, targets):
        if self.grid_mask:
            imgs = self.grid_mask(imgs)
        feat_list = self.img_backbone(imgs)
        feat_list = self.img_neck(feat_list)
        return feat_list

    def forward(self, images, batched_inputs):
        Ks, scale_ratios = self.scale_intrinsics(images, batched_inputs)    # Rescale the camera intrsincis based on input image resolution change.
        masks =  self.generate_mask(images, batched_inputs)  # Generate image mask for detr. masks shape: (b, h, w)
        pad_img_resolution = (images.tensor.shape[3],images.tensor.shape[2])

        feat_list = self.extract_feat(imgs = images.tensor, targets = batched_inputs)

        self.petr_head(feat_list, Ks, scale_ratios, masks, batched_inputs, pad_img_resolution)

    def scale_intrinsics(self, images, batched_inputs):
        height_im_scales_ratios = np.array([im.shape[1] / info['height'] for (info, im) in zip(batched_inputs, images)])
        width_im_scales_ratios = np.array([im.shape[2] / info['width'] for (info, im) in zip(batched_inputs, images)])
        Ks = copy.deepcopy(np.array([ele['K'] for ele in batched_inputs])) # Left shape: (B, 3, 3)
        
        Ks[:, 0, :] = Ks[:, 0, :] * height_im_scales_ratios[:, None]
        Ks[:, 1, :] = Ks[:, 1, :] * width_im_scales_ratios[:, None]
        Ks = torch.Tensor(Ks).cuda()
        scale_ratios = torch.Tensor(np.stack((height_im_scales_ratios, width_im_scales_ratios), axis = 0)).cuda()   # Left shape: (B, 2)
        return Ks, scale_ratios
    
    def generate_mask(self, images, batched_inputs):
        bs, _, mask_h, mask_w = images.tensor.shape
        masks = images.tensor.new_ones((bs, mask_h, mask_w))

        for idx, batched_input in enumerate(batched_inputs):
            ori_img_h, ori_img_w = batched_input['height'], batched_input['width']
            masks[idx][:ori_img_h, :ori_img_w] = 0

        return masks

def backbone_cfgs(backbone_name):
    cfgs = dict(
        ResNet50 = dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(2, 3,),
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d', requires_grad=False),
            norm_eval=True,
            style='caffe',
            with_cp=True,
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
            pretrained = 'MODEL/resnet50_msra-5891d200.pth',
        )
    )

    return cfgs[backbone_name]

def neck_cfgs(neck_name):
    cfgs = dict(
        CPFPN = dict(
            type='CPFPN',
            in_channels=[1024, 2048],
            out_channels=256,
            num_outs=2,
        )
    )

    return cfgs[neck_name]

class PETR_HEAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.LID = cfg.MODEL.DETECTOR3D.PETR.DEPTH_LID
        self.depth_num = 64
        self.embed_dims = 256
        self.position_dim = 3 * self.depth_num
        self.position_range = [-50, 50, -2, 4, 0, 100]  # (x_min, x_max, y_min, y_max, z_min, z_max). x: right, y: down, z: inwards

        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

    def position_embeding(self, feat, Ks, masks, batched_inputs, pad_img_resolution):
        eps = 1e-5
        pad_w, pad_h = pad_img_resolution
        B, _, H, W = feat.shape
        coords_h = torch.arange(H, device=feat.device).float() * pad_h / H
        coords_w = torch.arange(W, device=feat.device).float() * pad_w / W
        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=feat.device).float()
            index_1 = index + 1
            bin_size = (self.position_range[5] - self.position_range[4]) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.position_range[4] + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=feat.device).float()
            bin_size = (self.position_range[5] - self.position_range[4]) / self.depth_num
            coords_d = self.position_range[4] + bin_size * index
        
        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)  # Last dimension format (u*d, v*d, d)
        coords = coords[None].repeat(B, 1, 1, 1, 1)[..., None] # Left shape: (B, w, h, d, 3, 1)
        inv_Ks = Ks.inverse()[:, None, None, None]   # Left shape: (B, 1, 1, 1, 3, 3)
        cam_coords = (inv_Ks @ coords).squeeze(-1)    # cam_coords: Coordinates in the camera view coordinate system. Shape: (B, w, h, d, 3)

        cam_coords[..., 0:1] = (cam_coords[..., 0:1] - self.position_range[0]) / (self.position_range[1] - self.position_range[0])
        cam_coords[..., 1:2] = (cam_coords[..., 1:2] - self.position_range[2]) / (self.position_range[3] - self.position_range[2])
        cam_coords[..., 2:3] = (cam_coords[..., 2:3] - self.position_range[4]) / (self.position_range[5] - self.position_range[4])
        invalid_coords_mask = (cam_coords > 1.0) | (cam_coords < 0.0)   # Left shape: (B, w, h, d, 3)

        invalid_coords_mask = invalid_coords_mask.flatten(-2).sum(-1) > (D * 0.5)   # Left shape: (B, w, h)
        invalid_coords_mask = masks | invalid_coords_mask.permute(0, 2, 1)  # Left shape: (B, h, w)
        cam_coords = cam_coords.permute(0, 3, 4, 2, 1).contiguous().view(B, -1, H, W)   # (B, D*3, H, W)
        cam_coords = inverse_sigmoid(cam_coords)
        coords_position_embeding = self.position_encoder(cam_coords)    # Left shape: (B, C, H, W)
        
        return coords_position_embeding, invalid_coords_mask


    def forward(self, feat_list, Ks, scale_ratios, masks, batched_inputs, pad_img_resolution):
        feat = feat_list[0] # Left shape: (B, C, feat_h, feat_w)
        masks = F.interpolate(masks[None], size=feat.shape[-2:])[0].to(torch.bool)
        pos_embed, _ = self.position_embeding(feat, Ks, masks, batched_inputs, pad_img_resolution)  # Left shape: (B, C, feat_h, feat_w)
        pdb.set_trace()