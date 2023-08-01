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
from mmcv.runner.base_module import BaseModule
from detectron2.data import MetadataCatalog

from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet3d.models import build_backbone
from mmdet.models import build_neck as mmdet_build_neck
from mmdet3d.models import build_neck as mmdet3d_build_neck
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models import build_loss

from cubercnn.util.grid_mask import GridMask
from cubercnn.modeling import neck
from cubercnn.util.util import Converter_key2channel
from cubercnn.util import torch_dist
from cubercnn import util as cubercnn_util
from cubercnn.modeling.detector3d.detr_transformer import build_detr_transformer
from cubercnn.modeling.detector3d.deformable_detr import build_deformable_transformer
from cubercnn.util.util import box_cxcywh_to_xyxy, generalized_box_iou 
from cubercnn.util.math_util import get_cuboid_verts
from cubercnn.modeling.detector3d.depthnet import DepthNet
from voxel_pooling.voxel_pooling_train import voxel_pooling_train
from cubercnn.modeling.detector3d.center_head import CENTER_HEAD

class DETECTOR_PETR(BaseModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_enabled = cfg.MODEL.DETECTOR3D.PETR.BACKBONE_FP16

        self.img_mean = torch.Tensor(self.cfg.MODEL.PIXEL_MEAN)  # (b, g, r)
        self.img_std = torch.Tensor(self.cfg.MODEL.PIXEL_STD)

        if self.cfg.MODEL.DETECTOR3D.PETR.GRID_MASK:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        else:
            self.grid_mask = False

        backbone_cfg = backbone_cfgs(cfg.MODEL.DETECTOR3D.PETR.BACKBONE_NAME, cfg)
        if 'EVA' in cfg.MODEL.DETECTOR3D.PETR.BACKBONE_NAME:
            assert len(cfg.INPUT.RESIZE_TGT_SIZE) == 2
            assert cfg.INPUT.RESIZE_TGT_SIZE[0] == cfg.INPUT.RESIZE_TGT_SIZE[1]
            backbone_cfg['img_size'] = cfg.INPUT.RESIZE_TGT_SIZE[0]
        
        self.img_backbone = build_backbone(backbone_cfg)
        self.img_backbone.init_weights()

        if cfg.MODEL.DETECTOR3D.PETR.USE_NECK:
            neck_cfg = neck_cfgs(cfg.MODEL.DETECTOR3D.PETR.NECK_NAME, cfg)
            if neck_cfg['type'] == 'SECONDFPN':
                self.img_neck = mmdet3d_build_neck(neck_cfg)
                petr_head_inchannel = sum(neck_cfg['out_channels'])
            else:
                self.img_neck = mmdet_build_neck(neck_cfg)
                petr_head_inchannel = neck_cfg['out_channels']
        else:
            petr_head_inchannel = self.img_backbone.embed_dim

        if cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
            self.center_head = CENTER_HEAD(cfg, in_channels = petr_head_inchannel)
        
        self.petr_head = PETR_HEAD(cfg, in_channels = petr_head_inchannel)

    @auto_fp16(apply_to=('imgs'), out_fp32=True)
    def extract_feat(self, imgs, batched_inputs):
        if self.grid_mask and self.training:   
            imgs = self.grid_mask(imgs)
        backbone_feat_list = self.img_backbone(imgs)
        if type(backbone_feat_list) == dict: backbone_feat_list = list(backbone_feat_list.values())
        
        if self.cfg.MODEL.DETECTOR3D.PETR.USE_NECK:
            feat = self.img_neck(backbone_feat_list)[0]
        else:
            assert 'EVA' in self.cfg.MODEL.DETECTOR3D.PETR.BACKBONE_NAME
            patch_size = 14
            feat_h, feat_w = self.cfg.INPUT.RESIZE_TGT_SIZE[0] // patch_size, self.cfg.INPUT.RESIZE_TGT_SIZE[1] // patch_size
            feat = backbone_feat_list
            feat = feat.permute(0, 2, 1).reshape(feat.shape[0], -1, feat_h, feat_w).contiguous() # Left shape: (B, C, H, W)
        return feat

    def forward(self, images, batched_inputs, glip_results, class_name_emb):
        Ks, scale_ratios = self.transform_intrinsics(images, batched_inputs)    # Transform the camera intrsincis based on input image augmentations.
        masks = self.generate_mask(images)  # Generate image mask for detr. masks shape: (b, h, w)
        pad_img_resolution = (images.tensor.shape[3],images.tensor.shape[2])
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
        #box2d_centers = (batched_inputs[batch_id]['instances']._fields['gt_featreso_box2d_center'] * self.cfg.MODEL.DETECTOR3D.PETR.DOWNSAMPLE_FACTOR).int().cpu().numpy()
        #for box_center in box2d_centers:
        #    cv2.circle(img, box_center, radius = 3, color = (0, 0, 255), thickness = -1)
        print("Whether flip: {}".format(batched_inputs[batch_id]['horizontal_flip_flag']))
        pdb.set_trace()'''
        
        feat = self.extract_feat(imgs = images.tensor, batched_inputs = batched_inputs)

        if self.cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
            center_head_out = self.center_head(feat, Ks, batched_inputs)
        else:
            center_head_out = None
        
        petr_out = self.petr_head(feat, glip_results, class_name_emb, Ks, scale_ratios, masks, batched_inputs,
            pad_img_resolution = pad_img_resolution, 
            ori_img_resolution = ori_img_resolution, 
            center_head_out = center_head_out,
        )
        
        return dict(
            petr_out = petr_out,
            center_head_out = center_head_out,
            Ks = Ks,
        )

    def loss(self, detector_out, batched_inputs, ori_img_resolution, novel_cls_gts = None):
        center_head_loss_dict = self.center_head.loss(detector_out['center_head_out'], detector_out['Ks'], batched_inputs)
        loss_dict = self.petr_head.loss(detector_out['petr_out'], batched_inputs, ori_img_resolution, novel_cls_gts)
        loss_dict.update(center_head_loss_dict)
        return loss_dict
    
    def inference(self, detector_out, batched_inputs, ori_img_resolution):
        inference_results = self.petr_head.inference(detector_out['petr_out'], batched_inputs, ori_img_resolution)
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

def backbone_cfgs(backbone_name, cfg):
    cfgs = dict(
        ResNet50 = dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(2, 3,),
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d', requires_grad=False),
            norm_eval=False,
            style='caffe',
            with_cp=True,
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
            pretrained = 'MODEL/resnet50_msra-5891d200.pth',
        ),
        VoVNet = dict(
            type='VoVNet', ###use checkpoint to save memory
            spec_name='V-99-eSE',
            norm_eval=False,
            frozen_stages=-1,
            input_ch=3,
            out_features=cfg.MODEL.DETECTOR3D.PETR.FEAT_LEVEL_IDXS, #('stage3', 'stage4', 'stage5'),
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
    )

    return cfgs[backbone_name]

def neck_cfgs(neck_name, cfg):
    cfgs = dict(
        CPFPN_Res50 = dict(
            type='CPFPN',
            in_channels=[1024, 2048],
            out_channels=256,
            num_outs=2,
        ),
        CPFPN_VoV = dict(
            type='CPFPN',
            in_channels=[512, 768, 1024],
            out_channels=256,
            num_outs=len(cfg.MODEL.DETECTOR3D.PETR.FEAT_LEVEL_IDXS),
        ),
        SECONDFPN = dict(
            type='SECONDFPN',
            in_channels=[256, 512, 768, 1024],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128]
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
            enc_layers=cfg.MODEL.DETECTOR3D.PETR.HEAD.ENC_NUM,
            dec_layers=cfg.MODEL.DETECTOR3D.PETR.HEAD.DEC_NUM,
            pre_norm=False,
        ),
        DEFORMABLE_TRANSFORMER = dict(
            d_model = 256,
            nhead = 8,
            num_encoder_layers = cfg.MODEL.DETECTOR3D.PETR.HEAD.ENC_NUM,
            num_decoder_layers = cfg.MODEL.DETECTOR3D.PETR.HEAD.DEC_NUM,
            dim_feedforward = 512,
            dropout = 0.1,
            activation = "relu",
            return_intermediate_dec=True,
            num_feature_levels=1,   # Sample on the BEV feature.
            dec_n_points=4,
            enc_n_points=4,
            two_stage=False,
            two_stage_num_proposals=cfg.MODEL.DETECTOR3D.PETR.NUM_QUERY,
            use_dab=False,
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


class PETR_HEAD(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        assert len(set(cfg.DATASETS.CATEGORY_NAMES)) == len(cfg.DATASETS.CATEGORY_NAMES)
        if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
            assert len(set(cfg.DATASETS.NOVEL_CLASS_NAMES)) == len(cfg.DATASETS.NOVEL_CLASS_NAMES)
            assert len(set(cfg.DATASETS.CATEGORY_NAMES + cfg.DATASETS.NOVEL_CLASS_NAMES)) == len(set(cfg.DATASETS.CATEGORY_NAMES)) + len(set(cfg.DATASETS.NOVEL_CLASS_NAMES))
            self.total_cls_num = len(cfg.DATASETS.CATEGORY_NAMES) + len(cfg.DATASETS.NOVEL_CLASS_NAMES)
        else:
            self.total_cls_num = len(cfg.DATASETS.CATEGORY_NAMES)
        self.num_query = cfg.MODEL.DETECTOR3D.PETR.NUM_QUERY

        self.downsample_factor = cfg.MODEL.DETECTOR3D.PETR.DOWNSAMPLE_FACTOR
        self.position_range = cfg.MODEL.DETECTOR3D.PETR.HEAD.POSITION_RANGE  # (x_min, x_max, y_min, y_max, z_min, z_max). x: right, y: down, z: inwards
        self.x_bound = [self.position_range[0], self.position_range[1], cfg.MODEL.DETECTOR3D.PETR.HEAD.GRID_SIZE[0]]
        self.y_bound = [self.position_range[2], self.position_range[3], cfg.MODEL.DETECTOR3D.PETR.HEAD.GRID_SIZE[1]]
        self.z_bound = [self.position_range[4], self.position_range[5], cfg.MODEL.DETECTOR3D.PETR.HEAD.GRID_SIZE[2]]
        self.d_bound = cfg.MODEL.DETECTOR3D.PETR.HEAD.D_BOUND

        self.embed_dims = 256
        self.uncern_range = cfg.MODEL.DETECTOR3D.PETR.HEAD.UNCERN_RANGE

        self.register_buffer('voxel_size', torch.Tensor([row[2] for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.register_buffer('voxel_coord', torch.Tensor([row[0] + row[2] / 2.0 for row in [self.x_bound, self.y_bound, self.z_bound]]))
        self.register_buffer('voxel_num', torch.LongTensor([(row[1] - row[0]) / row[2]for row in [self.x_bound, self.y_bound, self.z_bound]]))

        if self.cfg.MODEL.DETECTOR3D.PETR.LID:
            depth_start, depth_end, depth_interval = self.d_bound
            depth_num = math.ceil((depth_end - depth_start) / depth_interval)
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (depth_end - depth_start) / (depth_num * (1 + depth_num))
            self.d_coords = depth_start + bin_size * index * index_1    # Left shape: (D,)

            coor_z_start, coor_z_end, coor_z_interval =  self.position_range[4], self.position_range[5], cfg.MODEL.DETECTOR3D.PETR.HEAD.GRID_SIZE[2]
            coor_z_num = math.ceil((coor_z_end - coor_z_start) / coor_z_interval)
            self.coor_z_num = coor_z_num
            coor_z_index  = torch.arange(start=0, end=coor_z_num, step=1).float()
            coor_z_index_1 = coor_z_index + 1
            coor_z_bin_size = (coor_z_end - coor_z_start) / (coor_z_num * (1 + coor_z_num))
            self.coor_z_coords = coor_z_start + coor_z_bin_size * coor_z_index * coor_z_index_1    # Left shape: (bev_z,)
        
        self.depth_channels = math.ceil((self.d_bound[1] - self.d_bound[0]) / self.d_bound[2])
        
        self.pos2d_generator = build_positional_encoding(dict(type='SinePositionalEncoding', num_feats=128, normalize=True))
        self.bev_pos2d_encoder = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

        self.depthnet = DepthNet(in_channels = self.in_channels, mid_channels = self.in_channels, context_channels = self.embed_dims, depth_channels = self.depth_channels, cfg = cfg)

        # Generating query heads
        self.lang_dim = cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        if self.cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
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
        transformer_cfg = transformer_cfgs(cfg.MODEL.DETECTOR3D.PETR.TRANSFORMER_NAME, cfg)
        transformer_cfg['cfg'] = cfg
        if cfg.MODEL.DETECTOR3D.PETR.TRANSFORMER_NAME == 'DETR_TRANSFORMER':
            self.transformer = build_detr_transformer(**transformer_cfg)
        elif cfg.MODEL.DETECTOR3D.PETR.TRANSFORMER_NAME == 'DEFORMABLE_TRANSFORMER':
            self.tgt_embed = nn.Embedding(self.num_query, self.embed_dims)
            self.transformer = build_deformable_transformer(**transformer_cfg)
        self.num_decoder = 6    # Keep this variable consistent with the detr configs.
        
        # Classification head
        self.num_reg_fcs = 2
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.cfg.MODEL.DETECTOR3D.PETR.HEAD.OV_CLS_HEAD:
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
        else:
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

        # For computing classification score based on image feature and class name feature.
        prior_prob = self.cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        log_scale = self.cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.LOG_SCALE
        self.dot_product_projection_text = nn.Linear(self.lang_dim, self.embed_dims, bias=True)
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(self.lang_dim), requires_grad=True)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # One-to-one macther
        self.cls_weight = cfg.MODEL.DETECTOR3D.PETR.HEAD.CLS_WEIGHT
        self.loc_weight = cfg.MODEL.DETECTOR3D.PETR.HEAD.LOC_WEIGHT
        self.dim_weight = cfg.MODEL.DETECTOR3D.PETR.HEAD.DIM_WEIGHT
        self.pose_weight = cfg.MODEL.DETECTOR3D.PETR.HEAD.POSE_WEIGHT
        self.det2d_l1_weight = cfg.MODEL.DETECTOR3D.PETR.HEAD.DET_2D_L1_WEIGHT
        self.det2d_iou_weight = cfg.MODEL.DETECTOR3D.PETR.HEAD.DET_2D_GIOU_WEIGHT
        
        matcher_cfg = matcher_cfgs(cfg.MODEL.DETECTOR3D.PETR.MATCHER_NAME, cfg)
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

        if cfg.MODEL.DETECTOR3D.OV_PROTOCOL: 
            matcher_2D_cfg = matcher_cfgs('HungarianAssigner2D', cfg)
            matcher_2D_cfg.update(dict(
                cls_weight = self.cls_weight,
                det2d_l1_weight = self.det2d_l1_weight,
                det2d_iou_weight = self.det2d_iou_weight,
                total_cls_num = self.total_cls_num,
                cfg = cfg,
            ))
            self.matcher_2D = build_assigner(matcher_2D_cfg)
        

        # Loss functions
        #self.cls_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        #self.mmdet_cls_loss = build_loss({'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 1.0})
        #self.reg_loss = build_loss({'type': 'L1Loss', 'loss_weight': 1.0})
        self.reg_loss = nn.L1Loss(reduction = 'none')
        self.iou_loss = build_loss({'type': 'GIoULoss', 'loss_weight': 0.0})
        
        self.init_weights()

    def init_weights(self):
        if self.cfg.MODEL.DETECTOR3D.PETR.TRANSFORMER_NAME == 'PETR_TRANSFORMER':
            self.transformer.init_weights()

        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        self.reference_points.weight.requires_grad = False

    def create_frustum(self, img_shape_after_aug, feat_shape):
        ogfW, ogfH = img_shape_after_aug
        fW, fH = feat_shape
        if self.cfg.MODEL.DETECTOR3D.PETR.LID:
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
        points = self.create_frustum(img_shape_after_aug, feat_shape)[None].expand(B, -1, -1, -1, -1).cuda()   # Left shape: (B, D, H, W, 3)
        
        points = torch.cat((points[..., :2] * points[..., 2:], points[..., 2:]), dim = -1)  # Left shape: (B, D, H, W, 3)
        points = points.unsqueeze(-1)  # Left shape: (1, D, H, W, 3, 1)
        Ks = Ks[:, None, None, None]    # Left shape: (B, 1, 1, 1, 3, 3)
        points = (Ks.inverse() @ points).squeeze(-1)    # Left shape: (B, D, H, W, 3)
        return points

    def forward(self, feat, glip_results, class_name_emb, Ks, scale_ratios, img_mask, batched_inputs, pad_img_resolution, ori_img_resolution, center_head_out = None):
        B = feat.shape[0]
        img_mask = F.interpolate(img_mask[None], size=feat.shape[-2:])[0].to(torch.bool)  # Left shape: (B, feat_h, feat_w)

        depth_and_feat = self.depthnet(feat, Ks)
        depth = depth_and_feat[:, :self.depth_channels].softmax(dim=1, dtype=depth_and_feat.dtype)  # Left shape: (B, depth_bin, H, W)
        img_feat_with_depth = depth.unsqueeze(1) * depth_and_feat[:, self.depth_channels:(self.depth_channels + self.embed_dims)].unsqueeze(2)  # Left shape: (B, C, D, H, W)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 3, 4, 1)    # Left shape: (B, D, H, W, C)
        geom_xyz = self.get_geometry(Ks, B, ori_img_resolution[0].int().cpu().numpy(), (img_mask.shape[2], img_mask.shape[1]))    # Left shape: (B, D, H, W, 3)
 
        if self.cfg.MODEL.DETECTOR3D.PETR.LID:
            geom_xyz = geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)
            geom_xy = (geom_xyz[..., :2] / self.voxel_size[:2]).int()    # Left shape: (B, D, H, W, 2)
            geom_z = (geom_xyz[..., 2:].unsqueeze(-1) >= self.coor_z_coords.cuda().view(1, 1, 1, 1, 1, -1)).int().sum(-1) - 1   # Left shape: (B, D, H, W, 1)
            geom_xyz = torch.cat((geom_xy, geom_z), dim = -1).int()   # Left shape: (B, D, H, W, 3)
        else:
            geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size).int()    # Left shape: (B, D, H, W, 3)
        geom_xyz[img_mask[:, None, :, :, None].expand_as(geom_xyz)] = -1   # Block feature from invalid image region.
        if self.cfg.MODEL.DETECTOR3D.PETR.LLS_SPARSE > 0:
            sparse_mask = depth < self.cfg.MODEL.DETECTOR3D.PETR.LLS_SPARSE # Left shape: (B, D, H, W)
            geom_xyz[sparse_mask.unsqueeze(-1).expand_as(geom_xyz)] = -1
        # Notably, the BEV feature shape should be (B, C bev_z, bev_x) rather than (B, C bev_y, bev_x).
        bev_feat = voxel_pooling_train(geom_xyz.contiguous(), img_feat_with_depth.contiguous(), self.voxel_num.cuda())   # Left shape: (B, C, bev_z, bev_x)
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
        
        # Prepare the class name embedding following the operations in GLIP.
        if self.cfg.MODEL.DETECTOR3D.PETR.HEAD.OV_CLS_HEAD:
            class_name_emb = class_name_emb[None].expand(B, -1, -1)
            class_name_emb = F.normalize(class_name_emb, p=2, dim=-1)
            dot_product_proj_tokens = self.dot_product_projection_text(class_name_emb / 2.0) # Left shape: (cls_num, L)
            dot_product_proj_tokens_bias = torch.matmul(class_name_emb, self.bias_lang) + self.bias0 # Left shape: (cls_num)

        reference_points = self.reference_points.weight[None].expand(B, -1, -1)  # Left shape: (B, num_query, 3) 
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))   # Left shape: (B, num_query, L) 
        tgt = self.tgt_embed.weight[None].expand(B, -1, -1) # Left shape: (B, num_query, L)

        if self.cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.USE_CENTER_PROPOSAL:
            if self.cfg.MODEL.DETECTOR3D.PETR.CENTER_PROPOSAL.TRAIN_GT_CENTER:
                tgt_conf_preds, tgt_xyz_preds = center_head_out['tgt_conf_preds'], center_head_out['tgt_xyz_preds']
                tgt_list = []
                for bs_idx in range(tgt.shape[0]):
                    center_conf_pred = tgt_conf_preds[bs_idx].detach()  # Left shape: (num_proposal, 1)
                    topk_center_xyz = tgt_xyz_preds[bs_idx].detach()  # Left shape: (num_proposal, 3)
                    center_conf_emb = self.center_conf_emb(center_conf_pred)   # Left shape: (num_proposal, L)
                    center_xyz_emb = self.center_xyz_emb(pos2posemb3d(topk_center_xyz)) # Left shape: (num_proposal, L)
                    center_proposal_num = center_conf_emb.shape[0]
                    bs_tgt = tgt[bs_idx].clone()    # Left shape: (num_query, L)
                    bs_tgt = torch.cat((bs_tgt[:center_proposal_num] + center_conf_emb + center_xyz_emb, bs_tgt[center_proposal_num:]), dim = 0)
                    tgt_list.append(bs_tgt)
                tgt = torch.stack(tgt_list, dim = 0)
            else:
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
            dec_cls_emb = self.cls_branches[lvl](outs_dec[lvl]) # Left shape: (B, num_query, emb_len) or (B, num_query, cls_num)
            dec_reg_result = self.reg_branches[lvl](outs_dec[lvl])  # Left shape: (B, num_query, reg_total_len)
            
            if self.cfg.MODEL.DETECTOR3D.PETR.HEAD.OV_CLS_HEAD:
                bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, self.num_query, 1)  
                dot_product_logit = (torch.matmul(dec_cls_emb, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias    # Left shape: (B, num_query, cls_num)
                if self.cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                    dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                    dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            else:
                dot_product_logit = dec_cls_emb.clone() # Left shape: (B, num_query, cls_num)
            outputs_classes.append(dot_product_logit)  # Left shape: (B, num_query, cls_num)
            outputs_regs.append(dec_reg_result)

        outputs_classes = torch.stack(outputs_classes, dim = 0)
        outputs_regs = torch.stack(outputs_regs, dim = 0)
        
        outputs_regs[..., self.reg_key_manager('loc')] = outputs_regs[..., self.reg_key_manager('loc')].sigmoid()

        outputs_regs[..., self.reg_key_manager('loc')][..., 0] = outputs_regs[..., self.reg_key_manager('loc')][..., 0] * \
            (self.position_range[1] - self.position_range[0]) + self.position_range[0]
        outputs_regs[..., self.reg_key_manager('loc')][..., 1] = outputs_regs[..., self.reg_key_manager('loc')][..., 1] * \
            (self.position_range[3] - self.position_range[2]) + self.position_range[2]
        if self.cfg.MODEL.DETECTOR3D.PETR.LID:
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
            if self.cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.CONFIDENCE_2D_THRE > 0:
                valid_mask = valid_mask & (max_cls_scores[bs] > self.cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.CONFIDENCE_2D_THRE)
            if self.cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.CONFIDENCE_3D_THRE > 0:
                valid_mask = valid_mask & (confs_3d[bs] > self.cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.CONFIDENCE_2D_THRE)
            bs_max_cls_scores = max_cls_scores[bs][valid_mask]
            bs_max_cls_idxs = max_cls_idxs[bs][valid_mask]
            bs_confs_3d = confs_3d[bs][valid_mask]
            bs_bbox_preds = bbox_preds[bs][valid_mask]
            
            # Prediction scores
            if self.cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.OUTPUT_CONFIDENCE == '2D':
                bs_scores = bs_max_cls_scores.clone()   # Left shape: (valid_query_num,)
            elif self.cfg.MODEL.DETECTOR3D.PETR.POST_PROCESS.OUTPUT_CONFIDENCE == '3D':
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
    
    def loss(self, detector_out, batched_inputs, ori_img_resolution, novel_cls_gts = None):
        all_cls_scores = detector_out['all_cls_scores'] # Left shape: (num_dec, B, num_query, cls_num)
        all_bbox_preds = detector_out['all_bbox_preds'] # Left shape: (num_dec, B, num_query, attr_num)

        batched_inputs = self.remove_instance_out_range(batched_inputs)
        
        num_dec_layers = all_cls_scores.shape[0]
        batched_inputs_list = [batched_inputs for _ in range(num_dec_layers)]
        dec_idxs = [dec_idx for dec_idx in range(num_dec_layers)]
        ori_img_resolutions = [ori_img_resolution for _ in range(num_dec_layers)]
        novel_cls_gts_list = [novel_cls_gts for _ in range(num_dec_layers)]
        
        decoder_loss_dicts = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, batched_inputs_list, novel_cls_gts_list, ori_img_resolutions, dec_idxs)[0]
        
        loss_dict = {}
        for ele in decoder_loss_dicts:
            loss_dict.update(ele)

        return loss_dict
        
    def loss_single(self, cls_scores, bbox_preds, batched_inputs, novel_cls_gts, ori_img_resolution, dec_idx):
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

        # One-to-One Matching between pseudo 2D labels and the remaining unmatched predictions.
        if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
            '''bbox2d_preds = []
            for bs, bs_bbox_preds in enumerate(bbox_preds_list):
                bs_pose_3x3 = rotation_6d_to_matrix(bs_bbox_preds[..., self.reg_key_manager('pose')])
                bs_corners_2d, bs_corners_3d = get_cuboid_verts(K = Ks[bs], box3d = torch.cat((bs_bbox_preds[..., self.reg_key_manager('loc')], bs_bbox_preds[..., self.reg_key_manager('dim')].exp()), dim = 1), R = bs_pose_3x3)
                bs_corners_2d = bs_corners_2d[:, :, :2] # Left shape: (num_box, 8, 2)
                bs_box_2d = bs_corners_2d.new_zeros(bs_corners_2d.shape[0], 2, 2)   # Left shape: (num_box, 2, 2)
                bs_u_min, bs_u_max = bs_corners_2d[..., 0].argmin(dim = 1), bs_corners_2d[..., 0].argmax(dim = 1)   # bs_u_min shape: (num_box,), bs_u_max shape: (num_box,)
                bs_v_min, bs_v_max = bs_corners_2d[..., 1].argmin(dim = 1), bs_corners_2d[..., 1].argmax(dim = 1)   # bs_v_min shape: (num_box,), bs_v_max shape: (num_box,)
                bs_box_2d[:, 0, 0:1] = bs_corners_2d[..., 0].gather(index = bs_u_min.unsqueeze(-1), dim = 1)    # The u coordinate of left up corner
                bs_box_2d[:, 0, 1:2] = bs_corners_2d[..., 1].gather(index = bs_v_min.unsqueeze(-1), dim = 1)    # The v coordinate of left up corner
                bs_box_2d[:, 1, 0:1] = bs_corners_2d[..., 0].gather(index = bs_u_max.unsqueeze(-1), dim = 1)    # The u coordinate of right bottom corner
                bs_box_2d[:, 1, 1:2] = bs_corners_2d[..., 1].gather(index = bs_v_max.unsqueeze(-1), dim = 1)    # The v coordinate of right bottom corner
                bbox2d_preds.append(bs_box_2d)'''
            proj3dcenter_preds = []
            for bs, bs_bbox_preds in enumerate(bbox_preds_list):
                proj_3d_center = (Ks[bs][None] @ bs_bbox_preds[..., self.reg_key_manager('loc')].unsqueeze(-1)).squeeze(-1) # Left shape: (num_box, 3)
                proj_3d_center = proj_3d_center[:, :2] / proj_3d_center[:, 2:]  # Left shape: (num_box, 2)
                proj_3d_center = proj_3d_center / ori_img_resolution[bs][None]  # Normalization. Left shape: (num_box, 2)
                proj3dcenter_preds.append(proj_3d_center)

            assign_results_2D = self.get_2D_matching(cls_scores_list, proj3dcenter_preds, ori_img_resolution, novel_cls_gts, assign_results)
            

        loss_dict = {'cls_loss_{}'.format(dec_idx): 0, 'loc_loss_{}'.format(dec_idx): 0, 'dim_loss_{}'.format(dec_idx): 0, 'pose_loss_{}'.format(dec_idx): 0}
        if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
            loss_dict.update({'proj3dcenter_loss_{}'.format(dec_idx): 0})

        num_pos = 0
        num_2d_pos = 0
        for bs, (pred_idxs, gt_idxs) in enumerate(assign_results):

            if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
                pred2d_idxs, gt2d_idxs = assign_results_2D[bs]
            
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
            Ks[1, :] = Ks[1, :] * ori_img_resolution[bs][1] / initial_h'''
            '''if novel_cls_gts[bs]['labels'].shape[0] != 0:
                pseudo_box2d_gt = novel_cls_gts[bs]['bbox'] # Left shape: (num_box, 4)
                pseudo_box2d_gt = torch.cat((ori_img_resolution[bs], ori_img_resolution[bs]), dim = 0)[None] * pseudo_box2d_gt
                pseudo_box2d_gt = pseudo_box2d_gt.int().cpu().numpy()
                for box in pseudo_box2d_gt:
                    cv2.rectangle(img, box[0:2], box[2:4], color=(255, 0, 0), thickness = 2)
                cv2.imwrite('vis.png', img)
                pdb.set_trace()
            else:
                continue'''
            '''corners_2d, corners_3d = get_cuboid_verts(K = Ks, box3d = torch.cat((bs_loc_preds, bs_dim_preds.exp()), dim = 1), R = bs_pose_gts)
            corners_2d = corners_2d[:, :, :2].detach()
            box2d = torch.cat((corners_2d.min(dim = 1)[0], corners_2d.max(dim = 1)[0]), dim = -1)
            for box in box2d:
                box = box.detach().cpu().numpy().astype(np.int32)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            for idx in range(bs_loc_preds.shape[0]):
                draw_3d_box(img, Ks.cpu().numpy(), torch.cat((bs_loc_preds[idx].detach().cpu(), bs_dim_preds[idx].exp().detach().cpu()), dim = 0).numpy(), bs_pose_preds[idx].detach().cpu().numpy())
            gt_boxes3D = batched_inputs[bs]['instances']._fields['gt_boxes3D']
            gt_proj_centers = gt_boxes3D[:, 0:2]    # The projected 3D centers have been resized.
            for gt_proj_center in gt_proj_centers:
                cv2.circle(img, gt_proj_center.cpu().numpy().astype(np.int32), radius = 3, color = (0, 0, 255), thickness = -1)
            cv2.imwrite('vis.png', img)
            pdb.set_trace()'''
            
            # Classification loss
            bs_cls_gts_onehot = bs_cls_gts.new_zeros((self.num_query, self.total_cls_num))   # Left shape: (num_query, num_cls)
            bs_valid_cls_gts_onehot = F.one_hot(bs_cls_gts, num_classes = self.total_cls_num) # Left shape: (num_gt, num_cls)
            bs_cls_gts_onehot[pred_idxs] = bs_valid_cls_gts_onehot
            if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
                bs_2d_cls_gts_onehot = F.one_hot(novel_cls_gts[bs]['labels'][gt2d_idxs], num_classes = self.total_cls_num)  # Left shape: (num_2d_gt, num_cls)
                bs_cls_gts_onehot[pred2d_idxs] = bs_2d_cls_gts_onehot.clone()
            loss_dict['cls_loss_{}'.format(dec_idx)] += self.cls_weight * torchvision.ops.sigmoid_focal_loss(bs_cls_scores, bs_cls_gts_onehot.float(), reduction = 'none').sum()
            
            # Localization loss
            loss_dict['loc_loss_{}'.format(dec_idx)] += self.loc_weight * (math.sqrt(2) * self.reg_loss(bs_loc_preds, bs_loc_gts)\
                                                        .sum(-1, keepdim=True) / bs_uncern_preds.exp() + bs_uncern_preds).sum()
            
            # Dimension loss
            loss_dict['dim_loss_{}'.format(dec_idx)] += self.dim_weight * self.reg_loss(bs_dim_preds, bs_dim_gts.log()).sum()

            # Pose loss
            loss_dict['pose_loss_{}'.format(dec_idx)] += self.pose_weight * (1 - so3_relative_angle(bs_pose_preds, bs_pose_gts, eps=1, cos_angle=True)).sum()

            # 2D IOU loss between 2D boxes converted from 3D box predictions and pseudo 2D box GTs.
            if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
                '''box2d_uvuv_preds = (bbox2d_preds[bs][pred2d_idxs] / ori_img_resolution[bs][None, None]).view(-1, 4)   # Left shape: (num_2d_gt, 4)
                box2d_uvuv_gts = novel_cls_gts[bs]['bbox'][gt2d_idxs]
                iou_loss = 1 - torch.diag(generalized_box_iou(box2d_uvuv_preds, box2d_uvuv_gts)) 
                iou_loss = F.relu(iou_loss - (1 - self.cfg.MODEL.DETECTOR3D.PETR.HEAD.DET2D_IOU_THRE))  # When IOU is bigger than DET2D_IOU_THRE, no loss punishment is applied.
                loss_dict['2d_iou_loss_{}'.format(dec_idx)] += self.det2d_iou_weight * iou_loss.sum()
                loss_dict['2d_l1_loss_{}'.format(dec_idx)] += self.det2d_l1_weight * self.reg_loss(box2d_uvuv_preds, box2d_uvuv_gts).sum()'''
                box2d_gt = novel_cls_gts[bs]['bbox'][gt2d_idxs].view(-1, 2, 2) # Left shape: (num_2d_gt, 2, 2)
                box_center_gt = box2d_gt.mean(dim = 1)  # Left shape: (num_2d_gt, 2)
                box2d_wh = (box2d_gt[:, 1, :] - box2d_gt[:, 0, :])  # Left shape: (num_2d_gt, 2)
                box2d_area = box2d_wh.sum(-1)    # Left shape: (num_gt)
                center_loss_threshold = self.cfg.MODEL.DETECTOR3D.PETR.HEAD.DET2D_AREA_FACTOR * box2d_area  # Left shape: (num_gt)
                loss_dict['proj3dcenter_loss_{}'.format(dec_idx)] +=  self.det2d_l1_weight * F.relu(self.reg_loss(proj3dcenter_preds[bs][pred2d_idxs], box_center_gt).sum(-1) - center_loss_threshold).sum()
            
            num_pos += gt_idxs.shape[0]
            if self.cfg.MODEL.DETECTOR3D.OV_PROTOCOL:
                num_2d_pos += gt2d_idxs.shape[0]
        
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

    def get_2D_matching(self, cls_scores_list, proj3dcenter_preds, ori_img_resolution, novel_cls_gts, assign_results):
        bs = len(cls_scores_list)
        ori_img_resolutions = torch.split(ori_img_resolution, 1, dim = 0)
        assign_results = multi_apply(self.get_2D_matching_single, cls_scores_list, proj3dcenter_preds, novel_cls_gts, ori_img_resolutions, assign_results)[0]
        return assign_results

    def get_2D_matching_single(self, cls_scores, proj3dcenter_pred, novel_cls_gt, ori_img_resolution, assign_result):
        assign_results = self.matcher_2D.assign(cls_scores, proj3dcenter_pred, novel_cls_gt, ori_img_resolution, assign_result)
        return assign_results

    def remove_instance_out_range(self, batched_inputs):
        for batch_id in range(len(batched_inputs)):
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
            batched_inputs[batch_id]['instances']._fields['gt_featreso_box2d_center'] =  batched_inputs[batch_id]['instances']._fields['gt_featreso_box2d_center'][instance_in_rang_flag]
            batched_inputs[batch_id]['instances']._fields['gt_featreso_2dto3d_offset'] =  batched_inputs[batch_id]['instances']._fields['gt_featreso_2dto3d_offset'][instance_in_rang_flag]
        return batched_inputs
        
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