import pdb
import os
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

from kitti_utils import kitti_utils

def prepare_mm_omni3d_kitti(kitti_root, omni3d_root):
    kitti_calib_folder = os.path.join(kitti_root, 'training/calib')
    kitti_lidar_folder = os.path.join(kitti_root, 'training/velodyne')

    omni3d_json_path = os.path.join(omni3d_root, 'Omni3D')
    mmOmni3d_json_path = os.path.join(omni3d_root, 'MM-Omni3D')
    mmOmni3d_lidar_path = os.path.join(omni3d_root, 'KITTI_object/training/velodyne')

    # Process lidar point and save it.
    omni3d_kitti_json_list = ['KITTI_train.json', 'KITTI_val.json', 'KITTI_test.json']
    for omni3d_kitti_json_name in omni3d_kitti_json_list:
        omni3d_kitti_json_file_path = os.path.join(omni3d_json_path, omni3d_kitti_json_name)
        with open(omni3d_kitti_json_file_path, 'r') as read_f:
            omni3d_kitti_json = json.load(read_f)
            omni3d_kitti_images = omni3d_kitti_json['images']
            print("Process {}...".format(omni3d_kitti_json_name))
            for cnt, omni3d_kitti_image_dict in enumerate(tqdm.tqdm(omni3d_kitti_images)):

                img_file_name = omni3d_kitti_image_dict['file_path'].rsplit('/')[-1]
                data_id = img_file_name.split('.')[0]

                calib_file_path = os.path.join(kitti_calib_folder, data_id + '.txt')
                calib = kitti_utils.Calibration(calib_file_path)

                lidar_file_path = os.path.join(kitti_lidar_folder, data_id + '.bin')
                lidar_points = kitti_utils.load_velo_scan(lidar_file_path)[:, :3]   # Left shape: (n_points, 3)
                lidar_points_ref = calib.project_velo_to_ref(lidar_points)  # Left shape: (n_points, 3)
                lidar_points_rect = calib.project_ref_to_rect(lidar_points_ref) # Left shape: (n_points, 3)
                lidar_points_uv, lidar_points_d = calib.project_rect_to_image(lidar_points_rect)    # lidar_points_uv shape: (n_points, 2), lidar_points_d shape: (n_points,)

                valid_point_mask = (lidar_points_d > 0) # Left shape: (n_points, )
                H, W = omni3d_kitti_image_dict['height'], omni3d_kitti_image_dict['width']
                valid_point_mask = valid_point_mask & (lidar_points_uv[:, 0] >= 0) & \
                    (lidar_points_uv[:, 0] <= W-1) & \
                    (lidar_points_uv[:, 1] >= 0) & \
                    (lidar_points_uv[:, 1] <= H-1)

                # Rectify 
                valid_lidar_points_rect = lidar_points_rect[valid_point_mask]
                valid_lidar_points_rect[:, 0] = valid_lidar_points_rect[:, 0] - calib.b_x
                valid_lidar_points_rect[:, 1] = valid_lidar_points_rect[:, 1] - calib.b_y

                # Save lidar points
                save_lidar_path = os.path.join(mmOmni3d_lidar_path, data_id + '.pcd.bin')
                valid_lidar_points_rect.astype(np.float32).tofile(save_lidar_path)

                # Add lidar point path to json
                lidar_json_path = os.path.join('KITTI_object/training/velodyne', data_id + '.pcd.bin') 
                omni3d_kitti_json['images'][cnt]['depth_file_path'] = lidar_json_path

        json_save_path = os.path.join(mmOmni3d_json_path, omni3d_kitti_json_name)
        with open(json_save_path, 'w') as write_f:
            json.dump(omni3d_kitti_json, write_f)
        print('Done!')

if __name__ == '__main__':
    kitti_root = '...'  # Change it to the root folder path of the downloaded KITTI dataset.
    omni3d_root = '...'  # Change it to the root folder path of saving the MM-Omni3D dataset.
    prepare_mm_omni3d_kitti(kitti_root = kitti_root, omni3d_root = omni3d_root)