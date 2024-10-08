import copy
from typing import Optional, List
import math
import pdb

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from mmdet.models.utils.transformer import inverse_sigmoid 
from mmdet.models.backbones.resnet import BasicBlock
from model.deformable_ops.modules import MSDeformAttn
from .depthnet import ASPP

def build_deformable_transformer(**kwargs):
    return DeformableTransformer(
        d_model=kwargs['d_model'],
        nhead=kwargs['nhead'],
        num_encoder_layers=kwargs['num_encoder_layers'],
        num_decoder_layers=kwargs['num_decoder_layers'],
        dim_feedforward=kwargs['dim_feedforward'],
        dropout=kwargs['dropout'],
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=kwargs['num_feature_levels'],
        dec_n_points=kwargs['dec_n_points'],
        enc_n_points=kwargs['enc_n_points'],
        two_stage=kwargs['two_stage'],
        two_stage_num_proposals=kwargs['two_stage_num_proposals'],
        use_dab=kwargs['use_dab'],
        cfg = kwargs['cfg'])

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False, cfg = None):
        super().__init__()

        self.cfg = cfg
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                        num_feature_levels, nhead, enc_n_points, cfg = cfg)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, cfg = cfg)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, cfg = cfg)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                            use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed, cfg = cfg)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, reg_branches = None, reference_points = None, \
        dataset_group_pred = None, reg_key_manager = None,  ori_img_resolution = None,):
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2).contiguous()                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2).contiguous()    # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, dataset_group_pred)
        
        # prepare input for decoder
        bs, _, c = memory.shape
        if self.use_dab:
            raise Exception("dab is not allowed.")
            reference_points = query_embed[..., self.d_model:].sigmoid() 
            tgt = query_embed[..., :self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=2)   # query_embed shape: (bs, num_query, 2), tgt shape: (bs, num_query, 2)
            init_reference_out = reference_points.clone()   # Left shape: (bs, num_query, 3)

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, 
                                            query_pos=query_embed if not self.use_dab else None, 
                                            src_padding_mask=mask_flatten, reg_branches = reg_branches, reg_key_manager = reg_key_manager, 
                                            ori_img_resolution = ori_img_resolution, dataset_group_pred = dataset_group_pred)

        return hs   # hs shape: (num_dec, bs, num_query, L)

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, cfg=None):
        super().__init__()
        self.cfg = cfg

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = AdaptiveLayerNorm(d_model, self.cfg.MODEL.DETECTOR3D.CENTER_PROPOSAL.ADAPT_LN, len(cfg.DATASETS.DATASET_ID_GROUP))
        self.norm2 = AdaptiveLayerNorm(d_model, self.cfg.MODEL.DETECTOR3D.CENTER_PROPOSAL.ADAPT_LN, len(cfg.DATASETS.DATASET_ID_GROUP))

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, dataset_group_pred = None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src, dataset_group_pred)

        # ffn
        src = self.forward_ffn(src)
        src = self.norm2(src, dataset_group_pred)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, cfg = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.cfg = cfg

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def _forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, dataset_group_pred = None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask, dataset_group_pred)

        return output

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, dataset_group_pred = None):
        if self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._forward, 
                src, 
                spatial_shapes, 
                level_start_index, 
                valid_ratios, 
                pos, 
                padding_mask,
                dataset_group_pred,
            )
        else:
            x = self._forward(
                src, 
                spatial_shapes, 
                level_start_index, 
                valid_ratios, 
                pos, 
                padding_mask,
                dataset_group_pred,
            )
        return x


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, cfg=None):
        super().__init__()
        self.cfg = cfg

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = AdaptiveLayerNorm(d_model, self.cfg.MODEL.DETECTOR3D.CENTER_PROPOSAL.ADAPT_LN, len(cfg.DATASETS.DATASET_ID_GROUP))

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = AdaptiveLayerNorm(d_model, self.cfg.MODEL.DETECTOR3D.CENTER_PROPOSAL.ADAPT_LN, len(cfg.DATASETS.DATASET_ID_GROUP))

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = AdaptiveLayerNorm(d_model, self.cfg.MODEL.DETECTOR3D.CENTER_PROPOSAL.ADAPT_LN, len(cfg.DATASETS.DATASET_ID_GROUP))

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, dataset_group_pred=None):
        if self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._forward, 
                tgt, 
                query_pos, 
                reference_points, 
                src, 
                src_spatial_shapes, 
                level_start_index, 
                src_padding_mask,
                dataset_group_pred,
            )
        else:
            x = self._forward(
                tgt, 
                query_pos, 
                reference_points, 
                src, 
                src_spatial_shapes, 
                level_start_index, 
                src_padding_mask,
                dataset_group_pred,
            )
        return x

    def _forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, dataset_group_pred=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(), tgt.transpose(0, 1).contiguous())[0].transpose(0, 1).contiguous()
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm1(tgt, dataset_group_pred)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt, dataset_group_pred)

        # ffn
        tgt = self.forward_ffn(tgt)
        tgt = self.norm3(tgt, dataset_group_pred)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, d_model=256, high_dim_query_update=False, no_sine_embed=False, cfg = None):
        super().__init__()
        self.cfg = cfg
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)


    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, reg_branches = None, reg_key_manager = None, 
                ori_img_resolution = None, dataset_group_pred = None):
        output = tgt
        if self.use_dab:
            assert query_pos is None
        bs = src.shape[0]
        
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            assert reference_points.shape[-1] == 3  # (x, y, z)
            reference_points_xz = torch.cat((reference_points[..., :1], reference_points[..., 2:]), dim = -1)   # Left shape: (B, num_query, 2)
            reference_points_xz_input = reference_points_xz[:, :, None] * src_valid_ratios[:, None]   # Make sure the reference points are in the valid image regions.

            if self.high_dim_query_update and lid != 0:
                assert False
                query_pos = query_pos + self.high_dim_query_proj(output)                 

            output = layer(output, query_pos, reference_points_xz_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, dataset_group_pred)
            
            if reg_branches is not None:
                tmp = reg_branches[lid](output)[..., reg_key_manager('loc')]   # Left shape: (B, num_query, 3)
                tmp_xz = torch.cat((tmp[..., :1], tmp[..., 2:]), dim = -1)  # Left shape: (B, num_query, 2)
                tmp_y = tmp[..., 1:2]   # Left shape: (B, num_query, 1)
                new_reference_points_xz = tmp_xz + inverse_sigmoid(reference_points_xz)
                new_reference_points = torch.cat((new_reference_points_xz[..., :1], tmp_y, new_reference_points_xz[..., 1:]), dim = -1).sigmoid()
                
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model, adaptive_norm=False, group_num=-1):
        super().__init__()
        self.d_model = d_model
        self.adaptive_norm = adaptive_norm
        self.group_num = group_num

        if not adaptive_norm:
            self.layernorm = nn.LayerNorm(d_model)
        else:
            self.layernorm = nn.LayerNorm(d_model, elementwise_affine=False)
            self.group_weight = nn.Embedding(group_num, d_model)
            self.group_bias = nn.Embedding(group_num, d_model)
            self.group_weight.weight.data.fill_(1.0)
            self.group_bias.weight.data.fill_(0.0)
    
    def forward(self, x, dataset_group_pred=None):
        if not self.adaptive_norm:
            return self.layernorm(x)
        else:
            x = self.layernorm(x)   # Left shape: (B, L, C)
            group_weight = self.group_weight.weight # Left shape: (group_num, C)
            group_bias = self.group_bias.weight # Left shape: (group_num, C)
            
            weight = (dataset_group_pred.unsqueeze(-1) * group_weight[None]).sum(dim = 1, keepdim = True)    # Left shape: (B, 1, C)
            bias = (dataset_group_pred.unsqueeze(-1) * group_bias[None]).sum(dim = 1, keepdim = True)   # Left shape: (B, 1, C)
            x = weight * x + bias
        return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos