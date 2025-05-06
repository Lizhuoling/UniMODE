import pdb
import os
import tqdm
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def prepare_mm_omni3d_sunrgbd(sunrgbd_root, omni3d_root):
    omni3d_json_path = sunrgbd_root
    mmOmni3d_json_path = omni3d_root

    omni3d_sunrgbd_json_list = ['SUNRGBD_train.json', 'SUNRGBD_val.json', 'SUNRGBD_test.json']
    for omni3d_sunrgbd_json_name in omni3d_sunrgbd_json_list:
        omni3d_sunrgbd_json_file_path = os.path.join(omni3d_json_path, omni3d_sunrgbd_json_name)
        with open(omni3d_sunrgbd_json_file_path, 'r') as read_f:
            omni3d_sunrgbd_json = json.load(read_f)
            omni3d_sunrgbd_images = omni3d_sunrgbd_json['images']
            print("Process {}...".format(omni3d_sunrgbd_json_name))

            for cnt, omni3d_sunrgbd_image_dict in enumerate(tqdm.tqdm(omni3d_sunrgbd_images)):

                img_file_name = omni3d_sunrgbd_image_dict['file_path'].rsplit('/')[-1]
                data_id = img_file_name.split('.')[0]
                
                depth_png_list = glob.glob(os.path.join(omni3d_root, omni3d_sunrgbd_image_dict['file_path'].rsplit('/', 2)[0], 'depth', '*' + 'png'))
                assert len(depth_png_list) == 1

                absolute_depth_file_path = depth_png_list[0]
                depth_file_path = os.path.join(omni3d_sunrgbd_image_dict['file_path'].rsplit('/', 2)[0], 'depth', absolute_depth_file_path.rsplit('/')[-1])

                img_file_path = os.path.join(omni3d_root, omni3d_sunrgbd_image_dict['file_path'])
                color_raw = o3d.io.read_image(img_file_path)
                
                depth_raw = o3d.io.read_image(absolute_depth_file_path)
                rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw)

                depth = np.array(rgbd_image.depth)

                pixel_u = np.arange(depth.shape[1])
                pixel_v = np.arange(depth.shape[0])
                pixel_u, pixel_v = np.meshgrid(pixel_u, pixel_v)

                valid_mask = depth > 0
                valid_depth = depth[valid_mask]
                valid_pixel_u = pixel_u[valid_mask]
                valid_pixel_v = pixel_v[valid_mask]

                valid_pixel_uv = np.stack((valid_pixel_u, valid_pixel_v), axis = -1)
                valid_pixel_uvd = np.concatenate((valid_pixel_uv * valid_depth[:, None], valid_depth[:, None]), axis = -1)
                K = np.array(omni3d_sunrgbd_image_dict['K'])
                inv_K = np.linalg.inv(K)
                valid_point_cloud = (inv_K[None] @ valid_pixel_uvd[..., None])[:, :, 0].astype(np.float32)
                
                lidar_path = depth_file_path.rsplit('.', 1)[0] + '.pcd.bin'
                save_lidar_path = os.path.join(omni3d_root, lidar_path)
                valid_point_cloud.tofile(save_lidar_path)

                omni3d_sunrgbd_json['images'][cnt]['depth_file_path'] = lidar_path
                
        json_save_path = os.path.join(mmOmni3d_json_path, omni3d_sunrgbd_json_name)
        with open(json_save_path, 'w') as write_f:
            json.dump(omni3d_sunrgbd_json, write_f)
        print('Done!')

if __name__ == '__main__':
    sunrgbd_root = '...'  # Change it to the root folder path of the downloaded SUN-RGBD dataset.
    omni3d_root = '...'  # Change it to the root folder path of saving the MM-Omni3D dataset.
    prepare_mm_omni3d_sunrgbd(sunrgbd_root = sunrgbd_root, omni3d_root = omni3d_root)