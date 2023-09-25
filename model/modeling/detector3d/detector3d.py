import pdb
import cv2
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

from pytorch3d.transforms import rotation_6d_to_matrix
from pytorch3d.transforms.so3 import so3_relative_angle
from detectron2.structures import Instances, Boxes
from detectron2.data import MetadataCatalog
from mmcv.runner.base_module import BaseModule

from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models import build_backbone
from mmdet.models import build_neck as mmdet_build_neck
from mmdet3d.models import build_neck as mmdet3d_build_neck
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models import build_loss

from model.util.grid_mask import GridMask
from model.modeling import neck
from model.util.util import Converter_key2channel
from model.util import torch_dist
from model import util as cubercnn_util
from model.modeling.detector3d.detr_transformer import build_detr_transformer
from model.modeling.detector3d.deformable_detr import build_deformable_transformer
from model.modeling.detector3d.point3d_transformer import build_point3d_transformer
from model.util.util import box_cxcywh_to_xyxy, generalized_box_iou 
from model.util.math_util import get_cuboid_verts
from model.modeling.detector3d.depthnet import DepthNet
from voxel_pooling.voxel_pooling_train import voxel_pooling_train
from model.modeling.detector3d.center_head import CENTER_HEAD
from model.modeling.backbone.dla import DLABackbone

class DETECTOR3D(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_enabled = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_FP16
        self.position_range = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.POSITION_RANGE

        self.img_mean = torch.Tensor(self.cfg.MODEL.PIXEL_MEAN)  # (b, g, r)
        self.img_std = torch.Tensor(self.cfg.MODEL.PIXEL_STD)

        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.GRID_MASK:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        else:
            self.grid_mask = False

        backbone_cfg = backbone_cfgs(cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME, cfg)
        if 'EVA' in cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME:
            assert len(cfg.INPUT.RESIZE_TGT_SIZE) == 2
            assert cfg.INPUT.RESIZE_TGT_SIZE[0] == cfg.INPUT.RESIZE_TGT_SIZE[1]
            backbone_cfg['img_size'] = cfg.INPUT.RESIZE_TGT_SIZE[0]
        
        if cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME == 'DLA34':
            self.img_backbone = DLABackbone(**backbone_cfg)
        else:
            self.img_backbone = build_backbone(backbone_cfg)
            self.img_backbone.init_weights()

        if cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.USE_NECK:
            neck_cfg = neck_cfgs(cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.NECK_NAME, cfg)
            if neck_cfg['type'] == 'SECONDFPN':
                self.img_neck = mmdet3d_build_neck(neck_cfg)
                detector3d_head_inchannel = sum(neck_cfg['out_channels'])
            else:
                self.img_neck = mmdet_build_neck(neck_cfg)
                detector3d_head_inchannel = neck_cfg['out_channels']
        else:
            detector3d_head_inchannel = self.img_backbone.embed_dim

        if cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
            self.center_head = CENTER_HEAD(cfg, in_channels = detector3d_head_inchannel)
        
        self.detector3d_head = DETECTOR3D_HEAD(cfg, in_channels = detector3d_head_inchannel)

    @auto_fp16(apply_to=('imgs'), out_fp32=True)
    def extract_feat(self, imgs, batched_inputs):
        if self.grid_mask and self.training:   
            imgs = self.grid_mask(imgs)
            
        backbone_feat_list = self.img_backbone(imgs)
        if type(backbone_feat_list) == dict: backbone_feat_list = list(backbone_feat_list.values())
        
        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.USE_NECK:
            feat = self.img_neck(backbone_feat_list)[0]
        else:
            assert 'EVA' in self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME
            patch_size = 14
            feat_h, feat_w = self.cfg.INPUT.RESIZE_TGT_SIZE[0] // patch_size, self.cfg.INPUT.RESIZE_TGT_SIZE[1] // patch_size
            feat = backbone_feat_list
            feat = feat.permute(0, 2, 1).reshape(feat.shape[0], -1, feat_h, feat_w).contiguous() # Left shape: (B, C, H, W)
        return feat

    def forward(self, images, batched_inputs):
        batched_inputs = self.remove_instance_out_range(batched_inputs)

        Ks, scale_ratios = self.transform_intrinsics(images, batched_inputs)    # Transform the camera intrsincis based on input image augmentations.
        masks = self.generate_mask(images)  # Generate image mask for detr. masks shape: (b, h, w)
        ori_img_resolution = torch.Tensor([(img.shape[2], img.shape[1]) for img in images]).to(images.device)   # (B, 2), width first then height
        
        # Visualize 3D gt centers using intrisics
        '''from cubercnn.vis.vis import draw_3d_box
        batch_id = 0
        img = batched_inputs[batch_id]['image'].permute(1, 2, 0).contiguous().cpu().numpy()
        K = Ks[batch_id]
        gts_2d = batched_inputs[batch_id]['instances']._fields['gt_boxes'].tensor
        for ele in gts_2d:
            box2d = ele.cpu().numpy().reshape(2, 2) # The box2d label has been rescaled to the input resolution.
            box2d = box2d.astype(np.int32)
            img = cv2.rectangle(img, box2d[0], box2d[1], (0, 255, 0))
        gts_3d = batched_inputs[batch_id]['instances']._fields['gt_boxes3D']
        for gt_cnt in range(gts_3d.shape[0]):
            gt_3d = gts_3d[gt_cnt]
            gt_center3d = gt_3d[6:9].cpu()
            gt_dim3d = gt_3d[3:6].cpu()
            gt_rot3d = batched_inputs[batch_id]['instances']._fields['gt_poses'][gt_cnt].cpu()
            draw_3d_box(img, K.cpu().numpy(), torch.cat((gt_center3d, gt_dim3d), dim = 0).numpy(), gt_rot3d.numpy()) d
        #box2d_centers = (batched_inputs[batch_id]['instances']._fields['gt_featreso_box2d_center'] * self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.DOWNSAMPLE_FACTOR).int().cpu().numpy()
        #for box_center in box2d_centers:
        #    cv2.circle(img, box_center, radius = 3, color = (0, 0, 255), thickness = -1)
        print("Whether flip: {}".format(batched_inputs[batch_id]['horizontal_flip_flag']))
        pdb.set_trace()'''
        
        feat = self.extract_feat(imgs = images.tensor, batched_inputs = batched_inputs)

        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
            center_head_out = self.center_head(feat, Ks, batched_inputs)
        else:
            center_head_out = None
        
        petr_out = self.detector3d_head(feat, Ks, scale_ratios, masks, batched_inputs, 
            ori_img_resolution = ori_img_resolution, 
            center_head_out = center_head_out,
        )
        detector_out = dict(
            petr_out = petr_out,
            center_head_out = center_head_out,
            Ks = Ks,
        )

        if self.training:
            return self.loss(detector_out, batched_inputs, ori_img_resolution)
        else:
            return self.inference(detector_out, batched_inputs, ori_img_resolution)

    def loss(self, detector_out, batched_inputs, ori_img_resolution):
        loss_dict = self.detector3d_head.loss(detector_out['petr_out'], batched_inputs, ori_img_resolution)
        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
            center_head_loss_dict = self.center_head.loss(detector_out['center_head_out'], detector_out['Ks'], batched_inputs)
            loss_dict.update(center_head_loss_dict)
        return loss_dict
    
    def inference(self, detector_out, batched_inputs, ori_img_resolution):
        inference_results = self.detector3d_head.inference(detector_out['petr_out'], batched_inputs, ori_img_resolution)
        return inference_results

    def transform_intrinsics(self, images, batched_inputs):
        height_im_scales_ratios = np.array([im.shape[1] / info['height'] for (info, im) in zip(batched_inputs, images)])
        width_im_scales_ratios = np.array([im.shape[2] / info['width'] for (info, im) in zip(batched_inputs, images)])
        scale_ratios = torch.Tensor(np.stack((width_im_scales_ratios, height_im_scales_ratios), axis = 1)).cuda()   # Left shape: (B, 2)
        Ks = copy.deepcopy(np.array([ele['K'] for ele in batched_inputs])) # Left shape: (B, 3, 3)
        
        for idx, batch_input in enumerate(batched_inputs):
            if 'horizontal_flip_flag' in batch_input.keys() and batch_input['horizontal_flip_flag']:
                Ks[idx][0][0] = -Ks[idx][0][0]
                Ks[idx][0][2] = batch_input['width'] - Ks[idx][0][2]
        
        Ks[:, 0, :] = Ks[:, 0, :] * width_im_scales_ratios[:, None] # Rescale intrinsics to the input image resolution.
        Ks[:, 1, :] = Ks[:, 1, :] * height_im_scales_ratios[:, None]
        Ks = torch.Tensor(Ks).cuda()
        return Ks, scale_ratios
    
    def generate_mask(self, images):
        bs, _, mask_h, mask_w = images.tensor.shape
        masks = images.tensor.new_ones((bs, mask_h, mask_w))

        for idx, batched_input in enumerate(range(bs)):
            valid_img_h, valid_img_w = images[idx].shape[1], images[idx].shape[2]
            masks[idx][:valid_img_h, :valid_img_w] = 0
        return masks

    def remove_instance_out_range(self, batched_inputs):
        for batch_id in range(len(batched_inputs)):
            if 'instances' in batched_inputs[batch_id].keys():
                gt_centers = batched_inputs[batch_id]['instances']._fields['gt_boxes3D'][:, 6:9]    # Left shape: (num_gt, 3)
                instance_in_rang_flag = (gt_centers[:, 0] > self.position_range[0]) & (gt_centers[:, 0] < self.position_range[1]) & \
                    (gt_centers[:, 1] > self.position_range[2]) & (gt_centers[:, 1] < self.position_range[3]) & \
                    (gt_centers[:, 2] > self.position_range[4]) & (gt_centers[:, 2] < self.position_range[5])   # Left shape: (num_gt)

                batched_inputs[batch_id]['instances']._fields['gt_classes'] = batched_inputs[batch_id]['instances']._fields['gt_classes'][instance_in_rang_flag]
                batched_inputs[batch_id]['instances']._fields['gt_boxes'].tensor = batched_inputs[batch_id]['instances']._fields['gt_boxes'].tensor[instance_in_rang_flag]
                batched_inputs[batch_id]['instances']._fields['gt_boxes3D'] = batched_inputs[batch_id]['instances']._fields['gt_boxes3D'][instance_in_rang_flag]
                batched_inputs[batch_id]['instances']._fields['gt_poses'] = batched_inputs[batch_id]['instances']._fields['gt_poses'][instance_in_rang_flag]
                batched_inputs[batch_id]['instances']._fields['gt_keypoints'] = batched_inputs[batch_id]['instances']._fields['gt_keypoints'][instance_in_rang_flag]
                batched_inputs[batch_id]['instances']._fields['gt_unknown_category_mask'] = batched_inputs[batch_id]['instances']._fields['gt_unknown_category_mask'][instance_in_rang_flag]
                if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
                    batched_inputs[batch_id]['instances']._fields['gt_featreso_box2d_center'] =  batched_inputs[batch_id]['instances']._fields['gt_featreso_box2d_center'][instance_in_rang_flag]
                    batched_inputs[batch_id]['instances']._fields['gt_featreso_2dto3d_offset'] =  batched_inputs[batch_id]['instances']._fields['gt_featreso_2dto3d_offset'][instance_in_rang_flag]
        return batched_inputs

def backbone_cfgs(backbone_name, cfg):
    cfgs = dict(
        ResNet50 = dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.FEAT_LEVEL_IDXS,
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d', requires_grad=False),
            norm_eval=False,
            style='caffe',
            with_cp=True,
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
            pretrained = 'MODEL/resnet50_msra-5891d200.pth',
        ),
        ResNet101 = dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.FEAT_LEVEL_IDXS,
            frozen_stages=1,
            norm_cfg=dict(type='BN2d', requires_grad=False),
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
            stage_with_dcn=(False, False, True, True),
            pretrained = 'MODEL/resnet101_caffe-3ad79236.pth',
        ),
        VoVNet = dict(
            type='VoVNet', ###use checkpoint to save memory
            spec_name='V-99-eSE',
            norm_eval=False,
            frozen_stages=-1,
            input_ch=3,
            out_features=cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.FEAT_LEVEL_IDXS,
            pretrained = 'MODEL/fcos3d_vovnet_imgbackbone_omni3d.pth',
        ),
        EVA_Base = dict(
            type = 'eva02_base_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE',
            pretrained = True,
            init_ckpt = 'MODEL/eva02_B_pt_in21k_medft_in21k_ft_in1k_p14.pt',
            use_mean_pooling = False,
        ),
        EVA_Large = dict(
            type = 'eva02_large_patch14_xattn_fusedLN_NaiveSwiGLU_subln_RoPE',
            pretrained = True,
            init_ckpt = 'MODEL/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt',
            use_mean_pooling = False,
        ),
        DLA34 = dict(
            cfg = None,
            input_shape = cfg.INPUT.RESIZE_TGT_SIZE,
            pretrained = True,
            DLA_TYPE = 'dla34',
            DLA_TRICKS = False,
        ),
        ConvNext_Base = dict(
            type = 'ConvNextBaseModel',
            pretrained = True,
            in_22k = True,
            out_indices = [0, 1, 2, 3],
        ),
    )

    return cfgs[backbone_name]

def neck_cfgs(neck_name, cfg):
    if cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME in ('ResNet50', 'ResNet101'):
        in_channels = [256, 512, 1024, 2048]
        upsample_strides = [0.25, 0.5, 1, 2]
        out_channels = [128, 128, 128, 128]
    elif cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME in ('VoVNet',):
        in_channels = [256, 512, 768, 1024]
        upsample_strides = [0.25, 0.5, 1, 2]
        out_channels = [128, 128, 128, 128]
    elif cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME in ('ConvNext_Base',):
        in_channels = [128, 256, 512, 1024]
        upsample_strides = [0.25, 0.5, 1, 2]
        out_channels = [128, 128, 128, 128]
    elif cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.BACKBONE_NAME in ('DLA34',):
        in_channels = [64, 128, 256, 512, 512]
        upsample_strides = [0.25, 0.5, 1, 2, 4]
        out_channels = [128, 128, 128, 128, 128]

    cfgs = dict(
        CPFPN_VoV = dict(
            type='CPFPN',
            in_channels=in_channels,
            out_channels=256,
            num_outs=len(cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.FEAT_LEVEL_IDXS),
        ),
        SECONDFPN = dict(
            type='SECONDFPN',
            in_channels=in_channels,
            upsample_strides=upsample_strides,
            out_channels=out_channels,
        ),
    )

    return cfgs[neck_name]

def transformer_cfgs(transformer_name, cfg):
    cfgs = dict(
        DETR_TRANSFORMER = dict(
            hidden_dim=256,
            dropout=0.1, 
            nheads=8,
            dim_feedforward=2048,
            enc_layers=cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.ENC_NUM,
            dec_layers=cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DEC_NUM,
            pre_norm=False,
        ),
        DEFORMABLE_TRANSFORMER = dict(
            d_model = 256,
            nhead = 8,
            num_encoder_layers = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.ENC_NUM,
            num_decoder_layers = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DEC_NUM,
            dim_feedforward = 512,
            dropout = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_DROPOUT,
            activation = "relu",
            return_intermediate_dec=True,
            num_feature_levels=1,   # Sample on the BEV feature.
            dec_n_points=4,
            enc_n_points=4,
            two_stage=False,
            two_stage_num_proposals=cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.NUM_QUERY,
            use_dab=False,
            cfg = cfg,
        ),
        Point3DTRANSFORMER = dict(
            d_model = 256,
            nhead = 8,
            num_decoder_layers = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DEC_NUM,
            dim_feedforward = 512,
            dropout = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_DROPOUT,
            activation = "relu",
            return_intermediate_dec=True,
            cfg = cfg,
        ),
    )

    return cfgs[transformer_name]

def matcher_cfgs(matcher_name, cfg):
    cfgs = dict(
        HungarianAssigner3D = dict(
            type='HungarianAssigner3D',
        ),
        HungarianAssigner2D = dict(
            type='HungarianAssigner2D',
        ),
    )
    return cfgs[matcher_name]


class DETECTOR3D_HEAD(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        if cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.CLS_MASK > 0:
            datasets_cls_dict = MetadataCatalog.get('omni3d_model').datasets_cls_dict
            total_cls_num = len(MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id)
            self.datasets_cls_mask = {}
            for dataset_id in datasets_cls_dict.keys():
                cls_mask = torch.zeros((total_cls_num,), dtype = torch.bool)
                cls_mask[datasets_cls_dict[dataset_id]] = True
                self.datasets_cls_mask[dataset_id] = cls_mask

        assert len(set(cfg.DATASETS.CATEGORY_NAMES)) == len(cfg.DATASETS.CATEGORY_NAMES)
        self.total_cls_num = len(cfg.DATASETS.CATEGORY_NAMES)
        self.num_query = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.NUM_QUERY

        self.downsample_factor = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.DOWNSAMPLE_FACTOR
        self.position_range = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.POSITION_RANGE  # (x_min, x_max, y_min, y_max, z_min, z_max). x: right, y: down, z: inwards
        self.x_bound = [self.position_range[0], self.position_range[1], cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.GRID_SIZE[0]]
        self.y_bound = [self.position_range[2], self.position_range[3], cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.GRID_SIZE[1]]
        self.z_bound = [self.position_range[4], self.position_range[5], cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.GRID_SIZE[2]]
        self.d_bound = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.D_BOUND

        self.embed_dims = 256
        self.uncern_range = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.UNCERN_RANGE

        self.register_buffer('voxel_size', torch.Tensor([row[2] for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.register_buffer('voxel_coord', torch.Tensor([row[0] + row[2] / 2.0 for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.register_buffer('voxel_num', torch.LongTensor([(row[1] - row[0]) / row[2]for row in [self.x_bound, self.y_bound, self.z_bound]]))

        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.LID_DEPTH:
            depth_start, depth_end, depth_interval = self.d_bound
            depth_num = math.ceil((depth_end - depth_start) / depth_interval)
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (depth_end - depth_start) / (depth_num * (1 + depth_num))
            self.d_coords = depth_start + bin_size * index * index_1    # Left shape: (D,)

        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.LID:
            coor_z_start, coor_z_end, coor_z_interval =  self.position_range[4], self.position_range[5], cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.GRID_SIZE[2]
            coor_z_num = math.ceil((coor_z_end - coor_z_start) / coor_z_interval)
            self.coor_z_num = coor_z_num
            coor_z_index  = torch.arange(start=0, end=coor_z_num, step=1).float()
            coor_z_index_1 = coor_z_index + 1
            coor_z_bin_size = (coor_z_end - coor_z_start) / (coor_z_num * (1 + coor_z_num))
            self.coor_z_coords = coor_z_start + coor_z_bin_size * coor_z_index * coor_z_index_1    # Left shape: (bev_z,)

        assert len(cfg.INPUT.RESIZE_TGT_SIZE) == 2
        img_shape_after_aug = cfg.INPUT.RESIZE_TGT_SIZE
        assert img_shape_after_aug[0] % self.downsample_factor == 0 and img_shape_after_aug[1] % self.downsample_factor == 0
        feat_shape = (img_shape_after_aug[0] // self.downsample_factor, img_shape_after_aug[1] // self.downsample_factor)
        self.register_buffer('frustum', self.create_frustum(cfg.INPUT.RESIZE_TGT_SIZE, feat_shape))
        
        self.depth_channels = math.ceil((self.d_bound[1] - self.d_bound[0]) / self.d_bound[2])
        
        self.pos2d_generator = build_positional_encoding(dict(type='SinePositionalEncoding', num_feats=128, normalize=True))
        self.bev_pos2d_encoder = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

        self.depthnet = DepthNet(in_channels = self.in_channels, mid_channels = self.in_channels, context_channels = self.embed_dims, depth_channels = self.depth_channels, cfg = cfg)

        # Generating query heads
        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
            self.center_conf_emb = nn.Linear(1, self.embed_dims)
            self.center_xyz_emb = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # Transformer
        transformer_cfg = transformer_cfgs(cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_NAME, cfg)
        transformer_cfg['cfg'] = cfg
        if cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_NAME == 'DETR_TRANSFORMER':
            self.transformer = build_detr_transformer(**transformer_cfg)
        elif cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_NAME == 'DEFORMABLE_TRANSFORMER':
            self.tgt_embed = nn.Embedding(self.num_query, self.embed_dims)
            self.transformer = build_deformable_transformer(**transformer_cfg)
        elif cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_NAME == 'Point3DTRANSFORMER':
            self.tgt_embed = nn.Embedding(self.num_query, self.embed_dims)
            self.transformer = build_point3d_transformer(**transformer_cfg)
        self.num_decoder = 6    # Keep this variable consistent with the detr configs.
        
        # Classification head
        self.num_reg_fcs = 2
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.total_cls_num))
        fc_cls = nn.Sequential(*cls_branch)
        self.cls_branches = nn.ModuleList([fc_cls for _ in range(self.num_decoder)])

        # Regression head
        reg_keys = ['loc', 'dim', 'pose', 'uncern']
        reg_chs = [3, 3, 6, 1]
        self.reg_key_manager = Converter_key2channel(reg_keys, reg_chs)
        self.code_size = sum(reg_chs)
        self.num_reg_fcs = 2
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_decoder)])

        # One-to-one macther
        self.cls_weight = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.CLS_WEIGHT
        self.loc_weight = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.LOC_WEIGHT
        self.dim_weight = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DIM_WEIGHT
        self.pose_weight = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.POSE_WEIGHT
        self.det2d_l1_weight = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DET_2D_L1_WEIGHT
        self.det2d_iou_weight = cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.DET_2D_GIOU_WEIGHT
        
        matcher_cfg = matcher_cfgs(cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.MATCHER_NAME, cfg)
        matcher_cfg.update(dict(total_cls_num = self.total_cls_num, 
                                cls_weight = self.cls_weight, 
                                loc_weight = self.loc_weight,
                                dim_weight = self.dim_weight,
                                pose_weight = self.pose_weight,
                                det2d_l1_weight = self.det2d_l1_weight,
                                det2d_iou_weight = self.det2d_iou_weight, 
                                uncern_range = self.uncern_range,
                                cfg = cfg,
        ))
        self.matcher = build_assigner(matcher_cfg)

        # Loss functions
        #self.cls_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        #self.mmdet_cls_loss = build_loss({'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 1.0})
        #self.reg_loss = build_loss({'type': 'L1Loss', 'loss_weight': 1.0})
        self.reg_loss = nn.L1Loss(reduction = 'none')
        self.iou_loss = build_loss({'type': 'GIoULoss', 'loss_weight': 0.0})
        
        self.init_weights()

    def init_weights(self):
        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.TRANSFORMER_NAME == 'PETR_TRANSFORMER':
            self.transformer.init_weights()

        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        self.reference_points.weight.requires_grad = False

    def create_frustum(self, img_shape_after_aug, feat_shape):
        ogfW, ogfH = img_shape_after_aug
        fW, fH = feat_shape
        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.LID_DEPTH:
            d_coords = self.d_coords.view(-1, 1, 1).expand(-1, fH, fW) # Left shape: (D, H, W)
        else:
            d_coords = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) # Left shape: (D, H, W)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((x_coords, y_coords, d_coords), -1) # Left shape: (D, H, W, 3)
        
        return frustum

    @torch.no_grad()
    def get_geometry(self, Ks, B, img_shape_after_aug, feat_shape):
        #points = self.create_frustum(img_shape_after_aug, feat_shape)[None].expand(B, -1, -1, -1, -1).cuda()   # Left shape: (B, D, H, W, 3)
        points = self.frustum[None].expand(B, -1, -1, -1, -1)   # Left shape: (B, D, H, W, 3)
        
        points = torch.cat((points[..., :2] * points[..., 2:], points[..., 2:]), dim = -1)  # Left shape: (B, D, H, W, 3)
        points = points.unsqueeze(-1)  # Left shape: (1, D, H, W, 3, 1)
        Ks = Ks[:, None, None, None]    # Left shape: (B, 1, 1, 1, 3, 3)
        points = (Ks.inverse() @ points).squeeze(-1)    # Left shape: (B, D, H, W, 3)
        return points

    def box_2d_emb(self, glip_bbox, Ks):
        '''
        Input:
        glip_bbox: The 2d boxes produced by GLIP. shape: (B, num_box, 4)
        Ks: Rescaled intrinsics. shape: (B, 3, 3)
        coords_d: Candidate depth values. shape: (depth_bin,)
        '''
        B, num_box, _ = glip_bbox.shape
        d_coors = torch.arange(*self.d_bound, dtype=torch.float).view(1, 1, 1, -1, 1).expand(B, num_box, 2, -1, 1).to(glip_bbox.device) # Left shape: (B, num_box, 2, D, 1)
        D = d_coors.shape[-2]
        glip_bbox = glip_bbox.view(B, num_box, 2, 1, 2).expand(-1, -1, -1, D, -1) # Left shape: (B, num_box, 2, D, 2)
        glip_bbox_3d = torch.cat((glip_bbox * d_coors, d_coors), dim = -1).unsqueeze(-1) # Left shape: (B, num_box, 2, D, 3, 1)
        Ks = Ks.view(B, 1, 1, 1, 3, 3) # Left shape: (B, 1, 1, 1, 3, 3)
        glip_bbox_3d = (Ks.inverse() @ glip_bbox_3d).view(B, num_box, 2 * D * 3) # Left shape: (B, num_box, 6 * D)
        glip_bbox_emb = self.glip_box_emb_mlp(glip_bbox_3d) # Left shape: (B, num_box, L)

        return glip_bbox_emb

    def forward(self, feat, Ks, scale_ratios, img_mask, batched_inputs, ori_img_resolution, center_head_out = None):
        B = feat.shape[0]
        img_mask = F.interpolate(img_mask[None], size=feat.shape[-2:])[0].to(torch.bool)  # Left shape: (B, feat_h, feat_w)

        depth_and_feat = self.depthnet(feat, Ks)
        depth = depth_and_feat[:, :self.depth_channels]
        depth = depth.softmax(dim=1, dtype=depth_and_feat.dtype)  # Left shape: (B, depth_bin, H, W)

        cam_feat = depth_and_feat[:, self.depth_channels:(self.depth_channels + self.embed_dims)]  # Left shape: (B, C, H, W)
        geom_xyz = self.get_geometry(Ks, B, ori_img_resolution[0].int().cpu().numpy(), (img_mask.shape[2], img_mask.shape[1]))    # Left shape: (B, D, H, W, 3)
 
        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.LID:
            geom_xyz = geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)
            geom_xy = (geom_xyz[..., :2] / self.voxel_size[:2]).int()    # Left shape: (B, D, H, W, 2)
            geom_z = (geom_xyz[..., 2:].unsqueeze(-1) >= self.coor_z_coords.cuda().view(1, 1, 1, 1, 1, -1)).int().sum(-1) - 1   # Left shape: (B, D, H, W, 1)
            geom_xyz = torch.cat((geom_xy, geom_z), dim = -1).int()   # Left shape: (B, D, H, W, 3)
        else:
            geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()    # Left shape: (B, D, H, W, 3)
        B, D, H, W, _ = geom_xyz.shape
        B_idx = torch.arange(start = 0, end = B, device = geom_xyz.device)
        D_idx = torch.arange(start = 0, end = D, device = geom_xyz.device)
        H_idx = torch.arange(start = 0, end = H, device = geom_xyz.device)
        W_idx = torch.arange(start = 0, end = W, device = geom_xyz.device)
        B_idx, D_idx, H_idx, W_idx = torch.meshgrid(B_idx, D_idx, H_idx, W_idx)
        HW_idx = H_idx * W + W_idx
        BDHW_idx = torch.stack((B_idx, D_idx, HW_idx), dim = -1)    # Left shape: (B, D, H, W, 3)
        geom_xyz = torch.cat((geom_xyz, BDHW_idx), dim = -1)    # Left shape: (B, D, H, W, 6). The first 3 numbers are voxel x, y, z. The remaning 3 numbers are b, d, hw.
        outrange_mask = (geom_xyz[..., 0] < 0) | (geom_xyz[..., 0] >= self.voxel_num[0]) | (geom_xyz[..., 1] < 0) | (geom_xyz[..., 1] >= self.voxel_num[1]) | (geom_xyz[..., 2] < 0) | (geom_xyz[..., 2] >= self.voxel_num[2])
        invalid_mask = img_mask.unsqueeze(1) | outrange_mask
        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.LLS_SPARSE > 0:
            invalid_mask = invalid_mask | (depth < self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.LLS_SPARSE) # Left shape: (B, D, H, W)
        valid_mask = ~invalid_mask
        geom_xyz = geom_xyz[valid_mask]    # Left shape: (num_proj, 6)
        
        # Notably, the BEV feature shape should be (B, C bev_z, bev_x) rather than (B, C bev_y, bev_x).
        voxel_feat = voxel_pooling_train(geom_xyz.int().contiguous(), cam_feat.contiguous(), depth.contiguous(), self.voxel_num)\
            .permute(0, 4, 1, 2, 3).contiguous()   # Left shape: (B, C, voxel_z, voxel_y, voxel_x)

        bev_feat = voxel_feat.sum(3)    # Left shape: (B, C, bev_z, bev_x)
        bev_mask = bev_feat.new_zeros(B, bev_feat.shape[2], bev_feat.shape[3])
        bev_feat_pos = self.pos2d_generator(bev_mask) # Left shape: (B, C, bev_z, bev_x)
        bev_feat_pos = self.bev_pos2d_encoder(bev_feat_pos) # Left shape: (B, C, bev_z, bev_x)
        
        # For vis
        '''depth_map = depth[0].argmax(0).cpu().numpy()
        plt.imshow(depth_map, cmap = 'magma_r')
        plt.savefig('depth.png')
        vis_bev_feat = bev_feat[0].mean(0).detach().cpu().numpy()
        plt.imshow(vis_bev_feat, cmap = 'magma_r')
        plt.savefig('vis.png')
        cv2.imwrite('ori.png', batched_inputs[0]['image'].permute(1, 2, 0).cpu().numpy())
        pdb.set_trace()'''

        reference_points = self.reference_points.weight[None].expand(B, -1, -1)  # Left shape: (B, num_query, 3) 
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))   # Left shape: (B, num_query, L) 
        tgt = self.tgt_embed.weight[None].expand(B, -1, -1) # Left shape: (B, num_query, L)

        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
            center_conf_pred = center_head_out['topk_center_conf'].detach()  # Left shape: (B, num_proposal, 1)
            topk_center_xyz = center_head_out['topk_center_xyz'].detach()    # Left shape: (B, num_proposal, 3)
            center_conf_emb = self.center_conf_emb(center_conf_pred)    # Left shape: (B, num_proposal, L)
            center_xyz_emb = self.center_xyz_emb(pos2posemb3d(topk_center_xyz))  # Left shape: (B, num_proposal, L)
            center_proposal_num = center_conf_emb.shape[1]
            center_initialized_tgt = tgt[:, :center_proposal_num].clone() + center_conf_emb + center_xyz_emb
            tgt = torch.cat((center_initialized_tgt, tgt[:, center_proposal_num:]), dim = 1) # Left shape: (B, num_query, L) 

        query_embeds = torch.cat((query_embeds, tgt), dim = -1)    # Left shape: (B, num_query, 2 * L)
        
        outs_dec, init_reference, inter_references, _, _ = self.transformer(srcs = [bev_feat], masks = [bev_mask.bool()], pos_embeds = [bev_feat_pos], query_embed = query_embeds, reg_branches = self.reg_branches, reference_points = reference_points, \
            reg_key_manager = self.reg_key_manager, ori_img_resolution = ori_img_resolution)

        
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_regs = []
        for lvl in range(outs_dec.shape[0]):
            if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.ITER_QUERY_UPDATE:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
            else:
                reference = init_reference
            reference = inverse_sigmoid(reference)

            dec_cls_emb = self.cls_branches[lvl](outs_dec[lvl]) # Left shape: (B, num_query, emb_len) or (B, num_query, cls_num)
            dec_reg_result = self.reg_branches[lvl](outs_dec[lvl])  # Left shape: (B, num_query, reg_total_len)
            dec_reg_result[..., self.reg_key_manager('loc')][..., 0:1] += reference[..., 0:1] # loc x
            dec_reg_result[..., self.reg_key_manager('loc')][..., 2:3] += reference[..., 2:3] # loc z
            dec_reg_result[...,self. reg_key_manager('loc')] = dec_reg_result[..., self.reg_key_manager('loc')].sigmoid()
  
            dot_product_logit = dec_cls_emb.clone() # Left shape: (B, num_query, cls_num)
            outputs_classes.append(dot_product_logit)  # Left shape: (B, num_query, cls_num)
            outputs_regs.append(dec_reg_result)

        outputs_classes = torch.stack(outputs_classes, dim = 0)
        outputs_regs = torch.stack(outputs_regs, dim = 0)

        outputs_regs[..., self.reg_key_manager('loc')][..., 0] = outputs_regs[..., self.reg_key_manager('loc')][..., 0] * \
            (self.position_range[1] - self.position_range[0]) + self.position_range[0]
        outputs_regs[..., self.reg_key_manager('loc')][..., 1] = outputs_regs[..., self.reg_key_manager('loc')][..., 1] * \
            (self.position_range[3] - self.position_range[2]) + self.position_range[2]
        if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.LID:
            pred_z = outputs_regs[..., self.reg_key_manager('loc')][..., 2].clone()
            pred_z = (pred_z * pred_z + pred_z) / 2 # Left shape: (num_dec, B, num_query)
            outputs_regs[..., self.reg_key_manager('loc')][..., 2] = pred_z * (self.position_range[5] - self.position_range[4]) + self.position_range[4]
        else:
            outputs_regs[..., self.reg_key_manager('loc')][..., 2] = outputs_regs[..., self.reg_key_manager('loc')][..., 2] * \
                (self.position_range[5] - self.position_range[4]) + self.position_range[4]
        
        outs = {
            'all_cls_scores': outputs_classes,  # outputs_classes shape: (num_dec, B, num_query, cls_num)
            'all_bbox_preds': outputs_regs, # outputs_regs shape: (num_dec, B, num_query, pred_attr_num)
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        
        return outs
    
    def inference(self, detector_out, batched_inputs, ori_img_resolution):
        cls_scores = detector_out['all_cls_scores'][-1]
        cls_scores = cls_scores.sigmoid() # Left shape: (B, num_query, cls_num)
        bbox_preds = detector_out['all_bbox_preds'][-1] # Left shape: (B, num_query, attr_num)
        B = cls_scores.shape[0]
        
        max_cls_scores, max_cls_idxs = cls_scores.max(-1) # max_cls_scores shape: (B, num_query), max_cls_idxs shape: (B, num_query)
        confs_3d_base_2d = torch.clamp(1 - bbox_preds[...,  self.reg_key_manager('uncern')].exp().squeeze(-1) / 10, min = 1e-3, max = 1)   # Left shape: (B, num_query)
        confs_3d = max_cls_scores * confs_3d_base_2d
        
        inference_results = []
        for bs in range(B):
            bs_instance = Instances(image_size = (batched_inputs[bs]['height'], batched_inputs[bs]['width']))
        
            valid_mask = torch.ones_like(max_cls_scores[bs], dtype = torch.bool)
            if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.CONFIDENCE_2D_THRE > 0:
                valid_mask = valid_mask & (max_cls_scores[bs] > self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.CONFIDENCE_2D_THRE)
            if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.CONFIDENCE_3D_THRE > 0:
                valid_mask = valid_mask & (confs_3d[bs] > self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.CONFIDENCE_2D_THRE)
            bs_max_cls_scores = max_cls_scores[bs][valid_mask]
            bs_max_cls_idxs = max_cls_idxs[bs][valid_mask]
            bs_confs_3d = confs_3d[bs][valid_mask]
            bs_bbox_preds = bbox_preds[bs][valid_mask]
            
            # Prediction scores
            if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.OUTPUT_CONFIDENCE == '2D':
                bs_scores = bs_max_cls_scores.clone()   # Left shape: (valid_query_num,)
            elif self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.POST_PROCESS.OUTPUT_CONFIDENCE == '3D':
                bs_scores = bs_confs_3d.clone()    # Left shape: (valid_query_num,)
            # Predicted classes
            bs_pred_classes = bs_max_cls_idxs.clone()
            # Predicted 3D boxes
            bs_pred_3D_center = bs_bbox_preds[..., self.reg_key_manager('loc')] # Left shape: (valid_query_num, 3)
            bs_pred_dims = bs_bbox_preds[..., self.reg_key_manager('dim')].exp()  # Left shape: (valid_query_num, 3)
            bs_pred_pose = bs_bbox_preds[..., self.reg_key_manager('pose')]   # Left shape: (valid_query_num, 6)
            bs_pred_pose = rotation_6d_to_matrix(bs_pred_pose.view(-1, 6)).view(-1, 3, 3)  # Left shape: (valid_query_num, 3, 3)
            bs_cube_3D = torch.cat((bs_pred_3D_center, bs_pred_dims), dim = -1) # Left shape: (valid_query_num, 6)
            Ks = torch.Tensor(np.array(batched_inputs[bs]['K'])).cuda() # Original K. Left shape: (3, 3)

            corners_2d, corners_3d = get_cuboid_verts(K = Ks, box3d = torch.cat((bs_pred_3D_center, bs_pred_dims), dim = 1), R = bs_pred_pose)
            corners_2d = corners_2d[:, :, :2]
            bs_pred_2ddet = torch.cat((corners_2d.min(dim = 1)[0], corners_2d.max(dim = 1)[0]), dim = -1)   # Left shape: (valid_query_num, 4). Predicted 2D box in the original image resolution.

            # For debug
            '''from cubercnn.vis.vis import draw_3d_box
            img = batched_inputs[0]['image'].permute(1, 2, 0).numpy()
            img = np.ascontiguousarray(img)
            img_reso = torch.cat((ori_img_resolution[bs], ori_img_resolution[bs]), dim = 0)
            initial_w, initial_h = batched_inputs[bs]['width'], batched_inputs[bs]['height']
            initial_reso = torch.Tensor([initial_w, initial_h, initial_w, initial_h]).to(img_reso.device)
            Ks[0, :] = Ks[0, :] * ori_img_resolution[bs][0] / initial_w
            Ks[1, :] = Ks[1, :] * ori_img_resolution[bs][1] / initial_h
            corners_2d, corners_3d = get_cuboid_verts(K = Ks, box3d = torch.cat((bs_pred_3D_center, bs_pred_dims), dim = 1), R = bs_pred_pose)
            corners_2d = corners_2d[:, :, :2].detach()
            box2d = torch.cat((corners_2d.min(dim = 1)[0], corners_2d.max(dim = 1)[0]), dim = -1)
            for box in box2d:
                box = box.detach().cpu().numpy().astype(np.int32)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            for idx in range(bs_pred_3D_center.shape[0]):
                draw_3d_box(img, Ks.cpu().numpy(), torch.cat((bs_pred_3D_center[idx].detach().cpu(), bs_pred_dims[idx].detach().cpu()), dim = 0).numpy(), bs_pred_pose[idx].detach().cpu().numpy())
            cv2.imwrite('vis.png', img)
            pdb.set_trace()'''

            # Obtain the projected 3D center on the original image resolution without augmentation
            proj_centers_3d = (Ks @ bs_pred_3D_center.unsqueeze(-1)).squeeze(-1)    # Left shape: (valid_query_num, 3)
            proj_centers_3d = proj_centers_3d[:, :2] / proj_centers_3d[:, 2:3]  # Left shape: (valid_query_num, 2)
            
            bs_instance.scores = bs_scores
            bs_instance.pred_boxes = Boxes(bs_pred_2ddet)
            bs_instance.pred_classes = bs_pred_classes
            bs_instance.pred_bbox3D = cubercnn_util.get_cuboid_verts_faces(bs_cube_3D, bs_pred_pose)[0]
            bs_instance.pred_center_cam = bs_cube_3D[:, :3]
            bs_instance.pred_dimensions = bs_cube_3D[:, 3:6]
            bs_instance.pred_pose = bs_pred_pose
            bs_instance.pred_center_2D = proj_centers_3d
            
            inference_results.append(dict(instances = bs_instance))
        
        return inference_results
    
    def loss(self, detector_out, batched_inputs, ori_img_resolution):
        all_cls_scores = detector_out['all_cls_scores'] # Left shape: (num_dec, B, num_query, cls_num)
        all_bbox_preds = detector_out['all_bbox_preds'] # Left shape: (num_dec, B, num_query, attr_num)
        
        num_dec_layers = all_cls_scores.shape[0]
        batched_inputs_list = [batched_inputs for _ in range(num_dec_layers)]
        dec_idxs = [dec_idx for dec_idx in range(num_dec_layers)]
        ori_img_resolutions = [ori_img_resolution for _ in range(num_dec_layers)]
        
        decoder_loss_dicts = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, batched_inputs_list, ori_img_resolutions, dec_idxs)[0]

        loss_dict = {}
        for ele in decoder_loss_dicts:
            loss_dict.update(ele)

        return loss_dict
        
    def loss_single(self, cls_scores, bbox_preds, batched_inputs, ori_img_resolution, dec_idx):
        '''
        Input:
            cls_scores: shape (B, num_query, cls_num)
            bbox_preds: shape (B, num_query, attr_num)
            batched_inputs: A list with the length of B. Each element is a dict containing labels.
        '''
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        device = cls_scores.device
        h_scale_ratios = [augmented_reso[1] / info['height'] for info, augmented_reso in zip(batched_inputs, ori_img_resolution)]

        Ks = torch.Tensor(np.array([ele['K'] for ele in batched_inputs])).cuda()
        height_im_scales_ratios = np.array([im_reso.cpu().numpy()[1] / info['height'] for (info, im_reso) in zip(batched_inputs, ori_img_resolution)])
        width_im_scales_ratios = np.array([im_reso.cpu().numpy()[0] / info['width'] for (info, im_reso) in zip(batched_inputs, ori_img_resolution)])
        scale_ratios = torch.Tensor(np.stack((width_im_scales_ratios, height_im_scales_ratios), axis = 1)).cuda()   # Left shape: (B, 2)
        Ks[:, 0, :] = Ks[:, 0, :] * scale_ratios[:, 0].unsqueeze(1)
        Ks[:, 1, :] = Ks[:, 1, :] * scale_ratios[:, 1].unsqueeze(1) # The Ks after rescaling
        Ks_list = [ele.squeeze(0) for ele in torch.split(Ks, 1, dim = 0)]

        # One-to-One Matching between 3D labels and predictions.
        assign_results = self.get_matching(cls_scores_list, bbox_preds_list, batched_inputs, ori_img_resolution, h_scale_ratios, Ks_list)

        loss_dict = {'cls_loss_{}'.format(dec_idx): 0, 'loc_loss_{}'.format(dec_idx): 0, 'dim_loss_{}'.format(dec_idx): 0, 'pose_loss_{}'.format(dec_idx): 0}

        num_pos = 0
        num_2d_pos = 0
        for bs, (pred_idxs, gt_idxs) in enumerate(assign_results):
            
            bs_cls_scores = cls_scores_list[bs] # Left shape: (num_all_query, cls_num)
            bs_bbox_preds = bbox_preds_list[bs][pred_idxs] # Left shape: (num_query, pred_attr_num)
            bs_loc_preds = bs_bbox_preds[..., self.reg_key_manager('loc')] # Left shape: (num_query, 3)
            bs_dim_preds = bs_bbox_preds[...,  self.reg_key_manager('dim')] # Left shape: (num_query, 3)
            bs_pose_preds = bs_bbox_preds[...,  self.reg_key_manager('pose')] # Left shape: (num_query, 6)
            bs_pose_preds = rotation_6d_to_matrix(bs_pose_preds)  # Left shape: (num_query, 3, 3)
            bs_uncern_preds = bs_bbox_preds[...,  self.reg_key_manager('uncern')] # Left shape: (num_query, 1)
            bs_uncern_preds = torch.clamp(bs_uncern_preds, min = self.uncern_range[0], max = self.uncern_range[1])
            
            bs_gt_instance = batched_inputs[bs]['instances']
            bs_cls_gts = bs_gt_instance.get('gt_classes').to(device)[gt_idxs] # Left shape: (num_gt,)
            bs_loc_gts = bs_gt_instance.get('gt_boxes3D')[..., 6:9].to(device)[gt_idxs] # Left shape: (num_gt, 3)
            bs_dim_gts = bs_gt_instance.get('gt_boxes3D')[..., 3:6].to(device)[gt_idxs] # Left shape: (num_gt, 3)
            bs_pose_gts = bs_gt_instance.get('gt_poses').to(device)[gt_idxs] # Left shape: (num_gt, 3, 3)

            # For debug
            '''from cubercnn.vis.vis import draw_3d_box
            img = batched_inputs[bs]['image'].permute(1, 2, 0).numpy()
            img = np.ascontiguousarray(img)
            img_reso = torch.cat((ori_img_resolution[bs], ori_img_resolution[bs]), dim = 0)
            initial_w, initial_h = batched_inputs[bs]['width'], batched_inputs[bs]['height']
            initial_reso = torch.Tensor([initial_w, initial_h, initial_w, initial_h]).to(img_reso.device)
            Ks = torch.Tensor(np.array(batched_inputs[bs]['K'])).cuda()
            Ks[0, :] = Ks[0, :] * ori_img_resolution[bs][0] / initial_w
            Ks[1, :] = Ks[1, :] * ori_img_resolution[bs][1] / initial_h
            # draw predictions
            #pred_corners_2d, pred_corners_3d = get_cuboid_verts(K = Ks, box3d = torch.cat((bs_loc_preds, bs_dim_preds.exp()), dim = 1), R = bs_pose_preds)
            #pred_corners_2d = pred_corners_2d[:, :, :2].detach()
            #pred_box2d = torch.cat((pred_corners_2d.min(dim = 1)[0], pred_corners_2d.max(dim = 1)[0]), dim = -1)
            #for box in pred_box2d:
            #    box = box.detach().cpu().numpy().astype(np.int32)
            #    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            #for idx in range(bs_loc_preds.shape[0]):
            #    draw_3d_box(img, Ks.cpu().numpy(), torch.cat((bs_loc_preds[idx].detach().cpu(), bs_dim_preds[idx].exp().detach().cpu()), dim = 0).numpy(), bs_pose_preds[idx].detach().cpu().numpy(), color=(0,255,0), thickness=2)
            # draw labels
            gt_corners_2d, gt_corners_3d = get_cuboid_verts(K = Ks, box3d = torch.cat((bs_loc_gts, bs_dim_gts), dim = 1), R = bs_pose_gts)
            gt_corners_2d = gt_corners_2d[:, :, :2].detach()
            gt_box2d = torch.cat((gt_corners_2d.min(dim = 1)[0], gt_corners_2d.max(dim = 1)[0]), dim = -1)
            #for box in gt_box2d:
            #    box = box.detach().cpu().numpy().astype(np.int32)
            #    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            for idx in range(bs_loc_gts.shape[0]):
                draw_3d_box(img, Ks.cpu().numpy(), torch.cat((bs_loc_gts[idx].detach().cpu(), bs_dim_gts[idx].detach().cpu()), dim = 0).numpy(), bs_pose_gts[idx].detach().cpu().numpy(), color=(0,0,255), thickness=2)
            gt_boxes3D = batched_inputs[bs]['instances']._fields['gt_boxes3D']
            #gt_proj_centers = gt_boxes3D[:, 0:2]    # The projected 3D centers have been resized.
            #for gt_proj_center in gt_proj_centers:
            #    cv2.circle(img, gt_proj_center.cpu().numpy().astype(np.int32), radius = 3, color = (0, 0, 255), thickness = -1)
            cv2.imwrite('vis.png', img)
            pdb.set_trace()'''
            
            # Classification loss
            bs_cls_gts_onehot = bs_cls_gts.new_zeros((self.num_query, self.total_cls_num))   # Left shape: (num_query, num_cls)
            bs_valid_cls_gts_onehot = F.one_hot(bs_cls_gts, num_classes = self.total_cls_num) # Left shape: (num_gt, num_cls)
            bs_cls_gts_onehot[pred_idxs] = bs_valid_cls_gts_onehot
            cls_loss_matrix = torchvision.ops.sigmoid_focal_loss(bs_cls_scores, bs_cls_gts_onehot.float(), reduction = 'none')  # Left shape: (num_query, num_cls)
            if self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.CLS_MASK > 0:
                valid_cls_mask = self.datasets_cls_mask[batched_inputs[bs]['dataset_id']].cuda()    # Left shape: (num_cls,) 
                cls_loss_matrix = cls_loss_matrix * valid_cls_mask[None] + self.cfg.MODEL.DETECTOR3D.TRANSFORMER_DETECTOR.HEAD.CLS_MASK * (cls_loss_matrix * ~valid_cls_mask[None])
            loss_dict['cls_loss_{}'.format(dec_idx)] += self.cls_weight * cls_loss_matrix.sum()
            
            # Localization loss
            loss_dict['loc_loss_{}'.format(dec_idx)] += self.loc_weight * (math.sqrt(2) * self.reg_loss(bs_loc_preds, bs_loc_gts)\
                                                        .sum(-1, keepdim=True) / bs_uncern_preds.exp() + bs_uncern_preds).sum()
            
            # Dimension loss
            loss_dict['dim_loss_{}'.format(dec_idx)] += self.dim_weight * self.reg_loss(bs_dim_preds, bs_dim_gts.log()).sum()

            # Pose loss
            loss_dict['pose_loss_{}'.format(dec_idx)] += self.pose_weight * (1 - so3_relative_angle(bs_pose_preds, bs_pose_gts, eps=100, cos_angle=True)).sum()
            
            num_pos += gt_idxs.shape[0]
        
        num_pos = torch_dist.reduce_mean(torch.Tensor([num_pos]).cuda()).item()
        torch_dist.synchronize()
    
        for key in loss_dict.keys():
            if 'cls_loss' in key:
                loss_dict[key] = loss_dict[key] / max(num_pos + num_2d_pos, 1)
            elif 'proj3dcenter_loss' in key:
                loss_dict[key] = loss_dict[key] / max(num_2d_pos, 1)
            else:
                loss_dict[key] = loss_dict[key] / max(num_pos, 1)
        
        return (loss_dict,)

    def get_matching(self, cls_scores_list, bbox_preds_list, batched_inputs, ori_img_resolution, h_scale_ratios, Ks_list):
        '''
        Description: 
            One-to-one matching for a single batch.
        '''
        bs = len(cls_scores_list)
        ori_img_resolution = torch.split(ori_img_resolution, 1, dim = 0)
        assign_results = multi_apply(self.get_matching_single, cls_scores_list, bbox_preds_list, batched_inputs, ori_img_resolution, h_scale_ratios, Ks_list)[0]
        return assign_results
        
    def get_matching_single(self, cls_scores, bbox_preds, batched_input, ori_img_resolution, h_scale_ratio, Ks):
        '''
        Description: 
            One-to-one matching for a single image.
        '''
        assign_result = self.matcher.assign(bbox_preds, cls_scores, batched_input, self.reg_key_manager, ori_img_resolution, h_scale_ratio, Ks)
        return assign_result

    def get_2D_matching(self, cls_scores_list, proj3dcenter_preds, ori_img_resolution, assign_results):
        bs = len(cls_scores_list)
        ori_img_resolutions = torch.split(ori_img_resolution, 1, dim = 0)
        assign_results = multi_apply(self.get_2D_matching_single, cls_scores_list, proj3dcenter_preds, ori_img_resolutions, assign_results)[0]
        return assign_results

    def get_2D_matching_single(self, cls_scores, proj3dcenter_pred, novel_cls_gt, ori_img_resolution, assign_result):
        assign_results = self.matcher_2D.assign(cls_scores, proj3dcenter_pred, novel_cls_gt, ori_img_resolution, assign_result)
        return assign_results
        
def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out