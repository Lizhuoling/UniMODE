import pdb
import cv2
import copy
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from pytorch3d.transforms import rotation_6d_to_matrix
from pytorch3d.transforms.so3 import so3_relative_angle

from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmdet3d.models import build_backbone
from mmdet.models import build_neck
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean

from cubercnn.util.grid_mask import GridMask
from cubercnn.modeling import neck
from cubercnn.util.util import Converter_key2channel
from cubercnn.util import torch_dist

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

        self.petr_head = PETR_HEAD(cfg, in_channels = neck_cfg['out_channels'])

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, imgs, targets):
        if self.grid_mask:
            imgs = self.grid_mask(imgs)
        feat_list = self.img_backbone(imgs)
        feat_list = self.img_neck(feat_list)
        return feat_list

    def forward(self, images, batched_inputs, glip_results, class_name_emb):
        Ks, scale_ratios = self.scale_intrinsics(images, batched_inputs)    # Rescale the camera intrsincis based on input image resolution change.
        masks =  self.generate_mask(images)  # Generate image mask for detr. masks shape: (b, h, w)
        pad_img_resolution = (images.tensor.shape[3],images.tensor.shape[2])

        # Visualize 3D gt centers using intrisics
        '''batch_id = 0
        img = batched_inputs[batch_id]['image'].permute(1, 2, 0).contiguous().cpu().numpy()
        K = Ks[batch_id]
        gts_2d = batched_inputs[batch_id]['instances']._fields['gt_boxes'].tensor
        for ele in gts_2d:
            box2d = ele.cpu().numpy().reshape(2, 2) # The box2d label has been rescaled to the input resolution.
            box2d = box2d.astype(np.int32)
            img = cv2.rectangle(img, box2d[0], box2d[1], (0, 255, 0))
        gts_3d = batched_inputs[batch_id]['instances']._fields['gt_boxes3D']
        for ele in gts_3d:
            center_3d = ele[6:9][:, None].to(K.device)
            center_3d_proj = (K @ center_3d)[:, 0]
            center_2d = center_3d_proj[:2] / center_3d_proj[2]
            #center_2d = ele[0:2].cpu().numpy()
            img = cv2.circle(img, (int(center_2d[0]), int(center_2d[1])), 4, (0, 0, 255), thickness = 2)
        cv2.imwrite('vis.png', img)
        pdb.set_trace()'''

        feat_list = self.extract_feat(imgs = images.tensor, targets = batched_inputs)
        petr_outs = self.petr_head(feat_list, glip_results, class_name_emb, Ks, scale_ratios, masks, batched_inputs, pad_img_resolution)

        return petr_outs

    def loss(self, detector_out, batched_inputs):
        loss_dict = self.petr_head.loss(detector_out, batched_inputs)
        return loss_dict

    def scale_intrinsics(self, images, batched_inputs):
        height_im_scales_ratios = np.array([im.shape[1] / info['height'] for (info, im) in zip(batched_inputs, images)])
        width_im_scales_ratios = np.array([im.shape[2] / info['width'] for (info, im) in zip(batched_inputs, images)])
        scale_ratios = torch.Tensor(np.stack((width_im_scales_ratios, height_im_scales_ratios), axis = 1)).cuda()   # Left shape: (B, 2)
        Ks = copy.deepcopy(np.array([ele['K'] for ele in batched_inputs])) # Left shape: (B, 3, 3)
        
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

def transformer_cfgs(transformer_name):
    cfgs = dict(
        PETR_TRANSFORMR = dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )
        )
    )

    return cfgs[transformer_name]

def matcher_cfgs(matcher_name):
    cfgs = dict(
        HungarianAssigner3D = dict(
            type='HungarianAssigner3D',
        )
    )
    return cfgs[matcher_name]


class PETR_HEAD(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.total_cls_num = len(cfg.DATASETS.CATEGORY_NAMES)
        self.num_query = cfg.MODEL.DETECTOR3D.PETR.NUM_QUERY
        self.glip_out_num = cfg.MODEL.GLIP_MODEL.MODEL.ATSS.DETECTIONS_PER_IMG
        self.random_init_query_num = self.num_query - self.glip_out_num
        assert self.random_init_query_num >= 0

        self.LID = cfg.MODEL.DETECTOR3D.PETR.DEPTH_LID
        self.depth_num = 64
        self.embed_dims = 256
        self.position_dim = 3 * self.depth_num
        self.position_range = [-50, 50, -2, 4, 0, 100]  # (x_min, x_max, y_min, y_max, z_min, z_max). x: right, y: down, z: inwards
        self.uncern_range = cfg.MODEL.DETECTOR3D.PETR.UNCERN_RANGE

        self.input_proj = Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        
        # 3D position coding head
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

        # Generating query heads
        self.lang_dim = cfg.MODEL.GLIP_MODEL.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        self.pos_conf_mlp = nn.Sequential(
                nn.Linear(5, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        self.glip_cls_emb_mlp = nn.Linear(self.lang_dim, self.embed_dims)
        self.glip_pos_emb_mlp = nn.Sequential(
                nn.Linear(self.depth_num * 6, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        self.glip_conf_emb = nn.Linear(1, self.embed_dims)
        if self.random_init_query_num > 0:
            self.reference_points = nn.Embedding(self.random_init_query_num, 3)
            self.unknown_cls_emb = nn.Embedding(1, self.embed_dims)
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )

        # Transformer
        transformer_cfg = transformer_cfgs(cfg.MODEL.DETECTOR3D.PETR.TRANSFORMER_NAME)
        self.transformer = build_transformer(transformer_cfg)

        self.num_decoder = 6    # Keep this variable consistent with the detr configs.

        # Classification head
        self.num_reg_fcs = 2
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.embed_dims))
        fc_cls = nn.Sequential(*cls_branch)
        self.cls_branches = nn.ModuleList([fc_cls for _ in range(self.num_decoder)])

        # Regression head
        reg_keys = ['loc', 'dim', 'pose', 'uncern']
        reg_chs = [3, 3, 4, 1]
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
        self.cls_weight = 2.0
        self.reg_weight = 0.25
        self.iou_weight = 0.0
        matcher_cfg = matcher_cfgs(cfg.MODEL.DETECTOR3D.PETR.MATCHER_NAME)
        matcher_cfg.update(dict(total_cls_num = self.total_cls_num, 
                                cls_weight = self.cls_weight, 
                                reg_weight = self.reg_weight,
                                iou_weight = self.iou_weight,
                                uncern_range = self.uncern_range,
                            ))
        self.matcher = build_assigner(matcher_cfg)

        # Loss functions
        self.cls_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        self.reg_loss = nn.L1Loss(reduction = 'none')
        
        self.init_weights()

    def init_weights(self):
        self.transformer.init_weights()
        if self.random_init_query_num > 0:
            nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    def produce_coords_d(self):
        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1).cuda().float()
            index_1 = index + 1
            bin_size = (self.position_range[5] - self.position_range[4]) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.position_range[4] + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1).cuda().float()
            bin_size = (self.position_range[5] - self.position_range[4]) / self.depth_num
            coords_d = self.position_range[4] + bin_size * index

        return coords_d

    def position_embeding(self, feat, coords_d, Ks, masks, pad_img_resolution):
        eps = 1e-5
        pad_w, pad_h = pad_img_resolution
        B, _, H, W = feat.shape
        coords_h = torch.arange(H, device=feat.device).float() * pad_h / H
        coords_w = torch.arange(W, device=feat.device).float() * pad_w / W
        
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

    def box_2d_emb(self, glip_bbox, Ks, coords_d):
        '''
        Input:
            glip_bbox: The 2d boxes produced by GLIP. shape: (B, num_box, 4)
            Ks: Rescaled intrinsics. shape: (B, 3, 3)
            coords_d: Candidate depth values. shape: (depth_bin,)
        '''
        B, num_box, _ = glip_bbox.shape
        depth_bin = coords_d.shape[0]
        glip_bbox  = glip_bbox.view(B, num_box, 1, 2, 2).expand(B, num_box, depth_bin, 2, 2)
        glip_bbox_3d = torch.cat((glip_bbox, coords_d.view(1, 1, depth_bin, 1, 1).expand(B, num_box, depth_bin, 2, 1)), dim = -1).unsqueeze(-1) # Left shape: (B, num_box, depth_bin, 2, 3, 1)
        inv_Ks = Ks.inverse()[:, None, None, None]  # Left shape: (B, 1, 1, 1, 3, 3)
        glip_bbox_3d = (inv_Ks @ glip_bbox_3d).squeeze(-1)    # Left shape: (B, num_box, depth_bin, 2, 3)
        glip_bbox_3d = glip_bbox_3d.view(B, num_box, -1)    # (B, num_box, depth_bin * 6)

        glip_bbox_emb = self.glip_pos_emb_mlp(glip_bbox_3d) # Left shape: (B, num_box, emb_len)

        return glip_bbox_emb

    def forward(self, feat_list, glip_results, class_name_emb, Ks, scale_ratios, masks, batched_inputs, pad_img_resolution):
        feat = self.input_proj(feat_list[0]) # Left shape: (B, C, feat_h, feat_w)
        B = feat.shape[0]

        # Prepare the class name embedding following the operations in GLIP.
        class_name_emb = class_name_emb[None].expand(B, -1, -1)
        class_name_emb = F.normalize(class_name_emb, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(class_name_emb / 2.0) # Left shape: (cls_num, L)
        dot_product_proj_tokens_bias = torch.matmul(class_name_emb, self.bias_lang) + self.bias0 # Left shape: (cls_num)

        masks = F.interpolate(masks[None], size=feat.shape[-2:])[0].to(torch.bool)
        coords_d = self.produce_coords_d()
        pos_embed, _ = self.position_embeding(feat, coords_d, Ks, masks, pad_img_resolution)  # Left shape: (B, C, feat_h, feat_w)

        glip_bbox_emb = self.box_2d_emb(glip_results['bbox'], Ks, coords_d) # Left shape: (B, num_box, L)
        glip_cls_emb = self.glip_cls_emb_mlp(glip_results['cls_emb'])  # Left shape: (B, num_box, L)
        glip_score_emb = self.glip_conf_emb(glip_results['scores'].unsqueeze(-1))   # Left shape: (B, num_box, L) 
        query_embeds = glip_bbox_emb + glip_cls_emb + glip_score_emb    # Left shape: (B, num_box, L) 
        if self.random_init_query_num > 0:
            random_reference_points = self.reference_points.weight
            random_query_embeds = self.query_embedding(pos2posemb3d(random_reference_points))
            random_query_embeds = random_query_embeds + self.unknown_cls_emb.weight # Left shape: (num_random_box, L) 
            query_embeds = torch.cat((query_embeds, random_query_embeds.unsqueeze(0).expand(B, -1, -1)), dim = 1)   # Left shape: (B, num_query, L)
        
        outs_dec, _ = self.transformer(feat, masks, query_embeds, pos_embed, self.reg_branches) # Left shape: (num_layers, bs, num_query, dim)
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_regs = []
        for lvl in range(outs_dec.shape[0]):
            dec_cls_emb = self.cls_branches[lvl](outs_dec[lvl]) # Left shape: (B, num_query, emb_len)
            dec_reg_result = self.reg_branches[lvl](outs_dec[lvl])  # Left shape: (B, num_query, reg_total_len)

            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, self.num_query, 1)  
            dot_product_logit = (torch.matmul(dec_cls_emb, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias    # Left shape: (B, num_query, cls_num)
            if self.cfg.MODEL.GLIP_MODEL.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            outputs_classes.append(dot_product_logit)  # Left shape: (B, num_query, cls_num)
            outputs_regs.append(dec_reg_result)

        outputs_classes = torch.stack(outputs_classes, dim = 0)
        outputs_regs = torch.stack(outputs_regs, dim = 0)

        outputs_regs[..., self.reg_key_manager('loc')] = outputs_regs[..., self.reg_key_manager('loc')].sigmoid()
        outputs_regs[..., self.reg_key_manager('loc')][..., 0] = outputs_regs[..., self.reg_key_manager('loc')][..., 0] * \
            (self.position_range[1] - self.position_range[0]) + self.position_range[0]
        outputs_regs[..., self.reg_key_manager('loc')][..., 1] = outputs_regs[..., self.reg_key_manager('loc')][..., 1] * \
            (self.position_range[3] - self.position_range[2]) + self.position_range[2]
        outputs_regs[..., self.reg_key_manager('loc')][..., 2] = outputs_regs[..., self.reg_key_manager('loc')][..., 2] * \
            (self.position_range[5] - self.position_range[4]) + self.position_range[4]

        outs = {
            'all_cls_scores': outputs_classes,  # outputs_classes shape: (num_dec, B, num_query, cls_num)
            'all_bbox_preds': outputs_regs, # outputs_regs shape: (num_dec, B, num_query, pred_attr_num)
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        
        return outs
    
    def loss(self, detector_out, batched_inputs):
        all_cls_scores = detector_out['all_cls_scores'] # Left shape: (num_dec, B, num_query, cls_num)
        all_bbox_preds = detector_out['all_bbox_preds'] # Left shape: (num_dec, B, num_query, attr_num)

        num_dec_layers = all_cls_scores.shape[0]
        batched_inputs_list = [batched_inputs for _ in range(num_dec_layers)]
        dec_idxs = [dec_idx for dec_idx in range(num_dec_layers)]

        decoder_loss_dicts = multi_apply(self.loss_single, all_cls_scores, all_bbox_preds, batched_inputs_list, dec_idxs)[0]

        loss_dict = {}
        for ele in decoder_loss_dicts:
            loss_dict.update(ele)

        return loss_dict
        
    def loss_single(self, cls_scores, bbox_preds, batched_inputs, dec_idx):
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
        
        assign_results = self.get_matching(cls_scores_list, bbox_preds_list, batched_inputs)

        loss_dict = {'cls_loss_{}'.format(dec_idx): 0, 'loc_loss_{}'.format(dec_idx): 0, 'dim_loss_{}'.format(dec_idx): 0, 'pose_loss_{}'.format(dec_idx): 0}
        num_pos = 0
        for bs, (pred_idxs, gt_idxs) in enumerate(assign_results):
            bs_cls_scores = cls_scores_list[bs][pred_idxs] # Left shape: (num_query, cls_num)
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
            
            # Classification loss
            bs_cls_cores_gts = F.one_hot(bs_cls_gts, num_classes = self.total_cls_num).float() # Left shape: (num_gt, 3)
            loss_dict['cls_loss_{}'.format(dec_idx)] += self.cls_weight * self.cls_loss(bs_cls_scores, bs_cls_cores_gts).sum()

            # Localization loss
            loss_dict['loc_loss_{}'.format(dec_idx)] += self.reg_weight * (math.sqrt(2) * self.reg_loss(bs_loc_preds, bs_loc_gts)\
                                                            .sum(-1, keepdim=True) / bs_uncern_preds.exp() + bs_uncern_preds).sum()

            # Dimension loss
            loss_dict['dim_loss_{}'.format(dec_idx)] += self.reg_weight * self.reg_loss(bs_dim_preds, bs_dim_gts.log()).sum()

            # Pose loss
            loss_dict['pose_loss_{}'.format(dec_idx)] += self.reg_weight * (1 - so3_relative_angle(bs_pose_preds, bs_pose_gts, eps=0.1, cos_angle=True)).sum()

            num_pos += gt_idxs.shape[0]
        
        num_pos = torch_dist.reduce_mean(torch.Tensor([num_pos]).cuda()).item()
        torch_dist.synchronize()
    
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / max(num_pos, 1)
        
        return (loss_dict,)

    def get_matching(self, cls_scores_list, bbox_preds_list, batched_inputs):
        '''
        Description: 
            One-to-one matching for a single batch.
        '''
        assign_results = multi_apply(self.get_matching_single, cls_scores_list, bbox_preds_list, batched_inputs)[0]
        return assign_results
        
    def get_matching_single(self, cls_scores, bbox_preds, batched_input):
        '''
        Description: 
            One-to-one matching for a single image.
        '''
        assign_result = self.matcher.assign(bbox_preds, cls_scores, batched_input, self.reg_key_manager)
        return assign_result

        
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
