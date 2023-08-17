import pdb
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.transformer import inverse_sigmoid 

def build_point3d_transformer(d_model, nhead, selfattn_block_x, selfattn_block_y, selfattn_block_z, num_decoder_layers, dim_feedforward, dropout, activation, return_intermediate_dec, cfg):
    return Point3DTransformer(
        d_model = d_model,
        nhead = nhead,
        selfattn_block_x = selfattn_block_x, 
        selfattn_block_y = selfattn_block_y, 
        selfattn_block_z = selfattn_block_z,
        num_decoder_layers = num_decoder_layers,
        dim_feedforward = dim_feedforward,
        dropout = dropout,
        activation = activation,
        return_intermediate_dec=return_intermediate_dec,
        cfg = cfg,
)

class Point3DTransformer(BaseModule):

    def __init__(self, d_model, nhead, selfattn_block_x, selfattn_block_y, selfattn_block_z, num_decoder_layers, dim_feedforward, dropout, activation, return_intermediate_dec, cfg):
        super(Point3DTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.cfg = cfg
        
        self.encoder = VoxelFeatEncoder(
            selfattn_block_x = selfattn_block_x, 
            selfattn_block_y = selfattn_block_x, 
            selfattn_block_z = selfattn_block_z,
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward = dim_feedforward, 
            dropout = dropout, 
            activation = activation,
            cfg = cfg,)

        decoder_layer = VoxelPoint3DDecoderLayer(d_model, nhead, dropout, cfg)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = VoxelPoint3DDecoder(decoder_layer, num_decoder_layers, norm=decoder_norm, return_intermediate=return_intermediate_dec, cfg=cfg)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, srcs, pos_embeds, query_embed, reg_branches, reference_points, reg_key_manager, ori_img_resolution):
        '''
        Input:
            srcs: 3D voxel feat with the shape of (B, C, voxel_z, voxel_y, voxel_x)
            pos_embeds: position embedding of srcs with the shape of (B, C, voxel_z, voxel_y, voxel_x)
            query_embed: shape: (B, num_query, 2*C)
        '''
        srcs = self.encoder(srcs, pos_embeds)   # Left shape: (B, C, voxel_z, voxel_y, voxel_x)

        query_pos, query = torch.split(query_embed, self.d_model , dim=2)   # query_embed shape: (bs, num_query, C), tgt shape: (bs, num_query, C)
        init_reference_out = reference_points.clone()
        
        hs, inter_references = self.decoder(
            query=query,
            key=None,
            value=srcs,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            reg_key_manager = reg_key_manager,)

        inter_references_out = inter_references
        
        return hs, init_reference_out, inter_references_out # Their shapes: (num_dec, B, num_query, C), (B, num_query, 3), (num_dec, B, num_query, 3)

class VoxelFeatEncoder(BaseModule):
    def __init__(self, selfattn_block_x, selfattn_block_y, selfattn_block_z, d_model, nhead, dim_feedforward, dropout, activation, cfg,):
        super(VoxelFeatEncoder, self).__init__()
        self.cfg = cfg

        self.block_x = selfattn_block_x
        self.block_y = selfattn_block_y
        self.block_z = selfattn_block_z
        self.block_len = self.block_x * self.block_y * self.block_z

        self.block_fc = nn.Sequential(
            nn.Linear(self.block_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.block_len * self.block_len),
        )

        self.channel_agg = nn.Linear(d_model, 1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, feat, feat_pos):
        '''
        feat shape: (B, C, voxel_z, voxel_y, voxel_x)
        feat_pos shape: (B, C, voxel_z, voxel_y, voxel_x)
        '''
        assert feat.ndim == 5 and feat_pos.ndim == 5
        B, C, voxel_z, voxel_y, voxel_x = feat.shape
        block_z, block_y, block_x = self.block_z, self.block_y, self.block_x

        ceil_voxel_z = math.ceil(voxel_z / block_z) * block_z
        ceil_voxel_y = math.ceil(voxel_y / voxel_z) * voxel_z
        ceil_voxel_x = math.ceil(voxel_x / block_x) * block_x
        pad_voxel_z = ceil_voxel_z - voxel_z
        pad_voxel_y = ceil_voxel_y - voxel_y
        pad_voxel_x = ceil_voxel_x - voxel_x
        
        #feat = feat + feat_pos
        pad_feat = F.pad(feat, (0, pad_voxel_z, 0, pad_voxel_y, 0, pad_voxel_x))    # Left shape: (B, C, ceil_voxel_z, ceil_voxel_y, ceil_voxel_x)  
        pad_feat = pad_feat.reshape(B, C, ceil_voxel_z // block_z, block_z, ceil_voxel_y // block_y, block_y, ceil_voxel_x // block_x, block_x)\
            .permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B * (ceil_voxel_z // block_z) * (ceil_voxel_y // block_y) * (ceil_voxel_x // block_x), block_z * block_y * block_x, C).contiguous()

        agg_feat = self.channel_agg(pad_feat).squeeze(-1)
        agg_conf = self.block_fc(agg_feat).sigmoid()  # Left shape: (num_block, block_len * block_len)
        agg_conf = agg_conf.view(agg_conf.shape[0], self.block_len, self.block_len)
        pad_feat = agg_conf @ pad_feat  # Left shape: (num_block, block_len, C)

        pad_feat = pad_feat.reshape(B, ceil_voxel_z // block_z, ceil_voxel_y // block_y, ceil_voxel_x // block_x, block_z, block_y, block_x, C).permute(0, 7, 1, 4, 2, 5, 3, 6)\
            .reshape(B, C, ceil_voxel_z, ceil_voxel_y, ceil_voxel_x)
        output = pad_feat[:, :, :voxel_z, :voxel_y, :voxel_x].contiguous()
        return output

class VoxelPoint3DDecoder(BaseModule):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, cfg = None):
        super(VoxelPoint3DDecoder, self).__init__()
        self.cfg = cfg

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                key,
                value,
                query_pos,
                reference_points,
                reg_branches,
                reg_key_manager):
        '''
        Input:
            query shape: (num_query, B, L)
            key: None
            value shape: (B, C, voxel_z, voxel_y, voxel_x)
            query_pos shape: (num_query, B, L)
            reference_points shape: (B, num_query, 3)
        '''
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points
            output = layer(
                output,
                key,
                value,
                query_pos = query_pos,
                reference_points = reference_points_input,
            )   # Left shape: (B, num_query, C)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)[..., reg_key_manager('loc')]
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)  # Left shape: (B, num_query, 3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)    # Left shape: (num_dec, B, num_query, C), right shape: (num_dec, B, num_query, 3)

        return output, reference_points # output shape: (B, num_query, C), reference_points shape: (B, num_query, 3)

class VoxelPoint3DDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 cfg = None,
        ):
        super(VoxelPoint3DDecoderLayer, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.cfg = cfg
        self.dropout = nn.Dropout(dropout)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attention_weights = nn.Linear(embed_dims, num_heads)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
            ):
        '''
        '''

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos
        
        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(bs, 1, num_query, self.num_heads).permute(0, 3, 1, 2).contiguous()   # Left shape: (B, num_head, 1, num_query)
        
        sample_feat = feature_sampling(value, reference_points)  # Left shape: (B, C, num_query)

        output = sample_feat.view(bs, self.num_heads, -1, num_query) * attention_weights.sigmoid()  # Left shape: (B, num_head, C // num_head, num_query)
        output = output.view(bs, -1, num_query).permute(0, 2, 1) # Left shape: (B, num_query, C)
        output = self.output_proj(output)

        return self.dropout(output) + inp_residual


def feature_sampling(voxel_feat, reference_points):
    '''
    voxel_feat shape: (B, C, voxel_z, voxel_y, voxel_x)
    reference_points shape: (B, num_query, 3)
    '''
    norm_reference_points = ((reference_points - 0.5) * 2)[:, None, None]    # Scale the range to (-1, 1). Left shape: (B, 1, 1, num_query, 3)
    sample_feat = F.grid_sample(voxel_feat, norm_reference_points)  # Left shape: (B, C, 1, 1, num_query)

    return sample_feat[:, :, 0, 0, :]

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])