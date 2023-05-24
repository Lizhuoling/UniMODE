import torch
import pdb

backbone_a = torch.load('backbone_b.pth', map_location = 'cpu')
backbone_b = torch.load('backbone_c.pth', map_location = 'cpu')

for key in backbone_a.keys():
    if key not in backbone_b.keys():
        pdb.set_trace()
    if (backbone_a[key] != backbone_b[key]).any():
        pdb.set_trace()
print('Done!')