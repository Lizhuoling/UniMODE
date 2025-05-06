import pdb
import os
import tqdm
import json
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def prepare_mm_omni3d_ar(ar_root, omni3d_root):
    omni3d_json_path = os.path.join(omni3d_root, 'Omni3D')
    mmOmni3d_json_path = os.path.join(omni3d_root, 'MM-Omni3D')
    mmOmni3d_lidar_path = os.path.join(omni3d_root, 'ARKitScenes/pointcloud')

    meta_data_csv_path = os.path.join(ar_root, 'raw/metadata.csv')
    meta_data_csv = pd.read_csv(meta_data_csv_path)

    omni3d_ar_json_list = ['ARKitScenes_train.json', 'ARKitScenes_val.json', 'ARKitScenes_test.json']
    for omni3d_ar_json_name in omni3d_ar_json_list:
        omni3d_ar_json_file_path = os.path.join(omni3d_json_path, omni3d_ar_json_name)
        with open(omni3d_ar_json_file_path, 'r') as read_f:
            omni3d_ar_json = json.load(read_f)
            omni3d_ar_images = omni3d_ar_json['images']
            print("Process {}...".format(omni3d_ar_json_name))
        for cnt, ar_image_dict in enumerate(tqdm.tqdm(omni3d_ar_images)):

            img_file_path = os.path.join(omni3d_root, ar_image_dict['file_path'])
            img = plt.imread(img_file_path)
            img_h, img_w, _ = img.shape

            _, split, seg_id, img_file_name = ar_image_dict['file_path'].rsplit('/', 3)
            data_id = img_file_name.rsplit('.', 1)[0]
            frame_id = data_id.rsplit('_', 1)[0]
            save_lidar_path = os.path.join(mmOmni3d_lidar_path, data_id + '.pcd.bin')

            if not os.path.exists(save_lidar_path):
                depth_path = os.path.join(ar_root, 'raw', split, seg_id, 'lowres_depth', "{}_{}.png".format(seg_id, frame_id))
                depth = cv2.imread(depth_path, -1) / 1000
                
                sky_dire = meta_data_csv.loc[meta_data_csv['video_id'] == int(seg_id)]['sky_direction'].item()
                if sky_dire == 'Up':
                    pass
                elif sky_dire == 'Left':
                    depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
                elif sky_dire == 'Down':
                    depth = cv2.rotate(depth, cv2.ROTATE_180)
                elif sky_dire == 'Right':
                    depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

                depth_h, depth_w = depth.shape

                K = np.array(ar_image_dict['K'])    # Left shape: (3, 3)
                scale_K = copy.deepcopy(K)
                scale_K[0, :] = scale_K[0, :] * depth_w / img_w
                scale_K[1, :] = scale_K[1, :] * depth_h / img_h

                depth_u = np.arange(depth_w)
                depth_v = np.arange(depth_h)
                depth_u, depth_v = np.meshgrid(depth_u, depth_v)

                mask = depth > 0
                depth = depth[mask]
                depth_u = depth_u[mask]
                depth_v = depth_v[mask]
                depth_uvd = np.stack((depth_u * depth, depth_v * depth, depth), axis = -1)
                point_xyz = (np.linalg.inv(scale_K) @ depth_uvd.T).T    # Left shape: (num_point, 3)

                # Save lidar points
                point_xyz.astype(np.float32).tofile(save_lidar_path)

            # Add lidar point path to json
            lidar_json_path = os.path.join('ARKitScenes/pointcloud', data_id + '.pcd.bin') 
            omni3d_ar_json['images'][cnt]['depth_file_path'] = lidar_json_path

        json_save_path = os.path.join(mmOmni3d_json_path, omni3d_ar_json_name)
        with open(json_save_path, 'w') as write_f:
            json.dump(omni3d_ar_json, write_f)
        print('Done!')

if __name__ == '__main__':
    ar_root = '...'  # Change it to the root folder path of the downloaded ARKitScenes dataset.
    omni3d_root = '...'  # Change it to the root folder path of saving the MM-Omni3D dataset.
    prepare_mm_omni3d_ar(
        ar_root = ar_root,
        omni3d_root = omni3d_root,
    )