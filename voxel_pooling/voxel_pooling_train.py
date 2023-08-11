import pdb

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import voxel_pooling_train_ext


class VoxelPoolingTrain(Function):

    @staticmethod
    def forward(ctx, geom_xyz: torch.Tensor, input_features: torch.Tensor, depth: torch.Tensor,
                voxel_num: torch.Tensor) -> torch.Tensor:
        """Forward function for `voxel pooling.

        Args:
            geom_xyz (Tensor): xyz coord for each voxel with the shape
                of [B, D, H, W, 3].
            input_features (Tensor): feature for each voxel with the
                shape of [B, C, H, W].
            depth shape: (B, D, H, W)
            voxel_num (Tensor): Number of voxels for each dim with the
                shape of [3], (voxel_x, voxel_y, voxel_z).

        Returns:
            Tensor: (B, C, bev_z, bev_x) bev feature map.
        """
        assert geom_xyz.is_contiguous()
        assert input_features.is_contiguous()
        assert depth.is_contiguous()

        ctx.mark_non_differentiable(geom_xyz)
        ctx.mark_non_differentiable(voxel_num)

        grad_input_features = torch.zeros_like(input_features)
        grad_depth = torch.zeros_like(depth)

        batch_size = input_features.shape[0]
        num_channels = input_features.shape[1]

        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1]).contiguous()  # Left shape: (B, D * H * W, 3)
        input_features = input_features.reshape(batch_size, num_channels, -1).contiguous()  # Left shape: (B, C, H * W)

        num_points = geom_xyz.shape[1]  # D * H * W
        num_features = input_features.shape[2]  # H * W
        num_depth = depth.shape[1]  # D

        output_features = input_features.new_zeros(batch_size, voxel_num[2], voxel_num[1], voxel_num[0], num_channels)
        
        # Save the position of bev_feature_map for each input point.
        voxel_pooling_train_ext.voxel_pooling_train_forward_wrapper(
            batch_size,
            num_points,
            num_features,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            num_depth,
            geom_xyz,
            input_features,
            depth,
            output_features,
        )
        # save for backward
        ctx.save_for_backward(grad_input_features, grad_depth, input_features.detach(), depth.detach(), geom_xyz, voxel_num)
        
        return output_features   # Left shape: (B, voxel_z, voxel_y, voxel_x, C)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_features):
        if not grad_output_features.is_contiguous():
            grad_output_features = grad_output_features.contiguous()

        (grad_input_features, grad_depth, input_features, depth, geom_xyz, voxel_num) = ctx.saved_tensors

        batch_size = input_features.shape[0]    # input_features shape: (B, C, H*W)
        num_points = geom_xyz.shape[1]  # geom_xyz shape: (B, D*H*W, 3)
        num_features = input_features.shape[2]
        num_channels = input_features.shape[1]
        num_depth = depth.shape[1]  # depth shape: (B, D, H, W)
        
        voxel_pooling_train_ext.voxel_pooling_train_backward_wrapper(
            batch_size,
            num_points,
            num_features,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            num_depth,
            geom_xyz,
            input_features,
            depth,
            grad_input_features,
            grad_depth,
            grad_output_features,
        )

        return None, grad_input_features, grad_depth, None
        


voxel_pooling_train = VoxelPoolingTrain.apply
