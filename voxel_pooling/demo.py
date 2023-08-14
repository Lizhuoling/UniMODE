import torch
import pdb

from voxel_pooling.voxel_pooling_train import voxel_pooling_train

iteration_num = 5

B = 2
C = 256
D = 100
H = 80
W = 100
voxel_x, voxel_y, voxel_z = 50, 1, 50

for iter in range(iteration_num):
    print("Iter: {}".format(iter))
    geom_xyz = torch.ones((B, D, H, W, 6), dtype = torch.int).cuda() # Left shape: (B, D, H, W, 6). The first 3 numbers are voxel x, y, z. The remaning 3 numbers are b, d, hw.
    geom_xyz[..., 0] = torch.randint(0, voxel_x, (B, D, H, W)).cuda()
    geom_xyz[..., 1] = torch.randint(0, voxel_y, (B, D, H, W)).cuda()
    geom_xyz[..., 2] = torch.randint(0, voxel_z, (B, D, H, W)).cuda()
    geom_xyz[..., 3] = torch.randint(0, B, (B, D, H, W)).cuda()
    geom_xyz[..., 4] = torch.randint(0, D, (B, D, H, W)).cuda()
    geom_xyz[..., 5] = torch.randint(0, H * W, (B, D, H, W)).cuda()
    geom_xyz = geom_xyz.view(-1, 6)


    cam_feat = torch.ones((B, C, H, W), dtype = torch.float32).cuda()
    cam_feat.requires_grad_(True)
    depth = torch.ones((B, D, H, W), dtype = torch.float32).softmax(dim = 1).cuda()
    depth.requires_grad_(True)
    voxel_num = torch.tensor((voxel_x, voxel_y, voxel_z), dtype = torch.int).cuda()

    voxel_feat = voxel_pooling_train(geom_xyz.int().contiguous(), cam_feat.contiguous(), depth.contiguous(), voxel_num).permute(0, 4, 1, 2, 3).contiguous()   # Left shape: (B, C, voxel_z, voxel_y, voxel_x)
    
    gt = voxel_feat.detach()
    loss = (voxel_feat - gt).sum()
    loss.backward()

pdb.set_trace()
