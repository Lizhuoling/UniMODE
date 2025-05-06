import pdb
import os
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

def prepare_mm_omni3d_nuscenes(nus_root, omni3d_root):
    omni3d_json_path = os.path.join(omni3d_root, 'Omni3D')
    mmOmni3d_json_path = os.path.join(omni3d_root, 'MM-Omni3D')
    mmOmni3d_lidar_path = os.path.join(omni3d_root, 'nuScenes/samples/LIDAR_TOP')

    nus = NuScenes(version = 'v1.0-trainval', dataroot = nus_root)
    
    # Map front camera image name to sample index
    frontimagename2sampleidx_map = {}
    print("Generate front camera image name to sample index mapping...")
    for sample in tqdm.tqdm(nus.sample):
        front_cam_sample_data_token = sample['data']['CAM_FRONT']
        front_cam_sample_data = nus.get('sample_data', front_cam_sample_data_token)
        front_cam_sample_img_path = front_cam_sample_data['filename']
        img_name = front_cam_sample_img_path.rsplit('/')[-1]
        frontimagename2sampleidx_map[img_name] = {}
        frontimagename2sampleidx_map[img_name]['sample_token'] = sample['token']
        frontimagename2sampleidx_map[img_name]['img_path'] = front_cam_sample_img_path
        frontimagename2sampleidx_map[img_name]['img_height'] = front_cam_sample_data['height']
        frontimagename2sampleidx_map[img_name]['img_width'] = front_cam_sample_data['width']
        cs_record_cam = nus.get('calibrated_sensor', front_cam_sample_data['calibrated_sensor_token'])
        pose_record_cam = nus.get('ego_pose', front_cam_sample_data['ego_pose_token'])
        frontimagename2sampleidx_map[img_name]['cs_record_cam'] = cs_record_cam
        frontimagename2sampleidx_map[img_name]['pose_record_cam'] = pose_record_cam

        lidar_sample_token = sample['data']['LIDAR_TOP']
        lidar_sample_data = nus.get('sample_data', lidar_sample_token)
        lidar_path = lidar_sample_data['filename']
        frontimagename2sampleidx_map[img_name]['lidar_path'] = lidar_path
        cs_record_lidar = nus.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
        pose_record_lidar = nus.get('ego_pose', lidar_sample_data['ego_pose_token'])
        frontimagename2sampleidx_map[img_name]['cs_record_lidar'] = cs_record_lidar
        frontimagename2sampleidx_map[img_name]['pose_record_lidar'] = pose_record_lidar
    
    # Process lidar point and save it.
    omni3d_nus_json_list = ['nuScenes_train.json', 'nuScenes_val.json', 'nuScenes_test.json']
    for omni3d_nus_json_name in omni3d_nus_json_list:
        omni3d_nus_json_file_path = os.path.join(omni3d_json_path, omni3d_nus_json_name)
        with open(omni3d_nus_json_file_path, 'r') as read_f:
            omni3d_nus_json = json.load(read_f)
            omni3d_nus_images = omni3d_nus_json['images']
            print("Process {}...".format(omni3d_nus_json_name))
            for cnt, omni3d_nus_image_dict in enumerate(tqdm.tqdm(omni3d_nus_images)):

                img_file_name = omni3d_nus_image_dict['file_path'].rsplit('/')[-1]

                lidar_path = frontimagename2sampleidx_map[img_file_name]['lidar_path']
                lidar_path = os.path.join(nus_root, lidar_path)
                lidar_points = LidarPointCloud.from_file(lidar_path).points # lidar_points shape: (4, num_points). Format: (x, y, z, tensity)
                lidar_points = lidar_points[:3, :].transpose(1, 0)  # Left shape: (num_points, 3). In the ego coordinate system.
                
                # Lidar to World Transformation
                lidar2ego_rotation = Quaternion(frontimagename2sampleidx_map[img_file_name]['cs_record_lidar']['rotation']).rotation_matrix # Left shape: (3, 3)
                lidar2ego_translation = np.array(frontimagename2sampleidx_map[img_file_name]['cs_record_lidar']['translation'])[..., None] # Left shape: (3, 1)
                lidar_ego2world_rotation = Quaternion(frontimagename2sampleidx_map[img_file_name]['pose_record_lidar']['rotation']).rotation_matrix # Left shape: (3, 3)
                lidar_ego2world_translation = np.array(frontimagename2sampleidx_map[img_file_name]['pose_record_lidar']['translation'])[..., None] # Left shape: (3, 1)
                lidar2world_rotation = lidar_ego2world_rotation @ lidar2ego_rotation   # Left shape: (3, 3)
                lidar2world_translation = lidar_ego2world_rotation @ lidar2ego_translation + lidar_ego2world_translation    # Left shape: (3, 1)
                # World to Cam Transformation
                ego2cam_rotation = Quaternion(frontimagename2sampleidx_map[img_file_name]['cs_record_cam']['rotation']).inverse.rotation_matrix # Left shape: (3, 3)
                ego2cam_translation = -ego2cam_rotation @ np.array(frontimagename2sampleidx_map[img_file_name]['cs_record_cam']['translation'])[..., None] # Left shape: (3, 1)
                cam_world2ego_rotation = Quaternion(frontimagename2sampleidx_map[img_file_name]['pose_record_cam']['rotation']).inverse.rotation_matrix # Left shape: (3, 3)
                cam_world2ego_translation = -cam_world2ego_rotation @ np.array(frontimagename2sampleidx_map[img_file_name]['pose_record_cam']['translation'])[..., None] # Left shape: (3, 1)
                world2cam_rotation = ego2cam_rotation @ cam_world2ego_rotation  # Left shape: (3, 3)
                world2cam_translation = ego2cam_rotation @ cam_world2ego_translation + ego2cam_translation  # Left shape: (3, 1)
                # Lidar to Cam Transformation
                lidar2cam_rotation = world2cam_rotation @ lidar2world_rotation  # Left shape: (3, 3)
                lidar2cam_translation = world2cam_rotation @ lidar2world_translation + world2cam_translation    # Left shape: (3, 1)
                # Camera intrinsics
                K = np.array(frontimagename2sampleidx_map[img_file_name]['cs_record_cam']['camera_intrinsic'])   # Left shape: (3, 3)

                # Remove invalid lidar points
                lidar_points_cam = lidar2cam_rotation @ lidar_points[..., None] + lidar2cam_translation    # Left shape: (num_points, 3, 1)
                lidar_points_uvd = (K @ lidar_points_cam)[:, :, 0] # Left shape: (num_points, 3)
                valid_point_mask = lidar_points_uvd[:, 2] > 0 # Left shape: (num_points,)
                lidar_points_uv = lidar_points_uvd[:, :2] / lidar_points_uvd[:, 2:] # Left shape: (num_points, 2)
                lidar_points_d = lidar_points_uvd[:, 2:]    # Left shape: (num_points, 1)
                H, W = frontimagename2sampleidx_map[img_file_name]['img_height'], frontimagename2sampleidx_map[img_file_name]['img_width']
                valid_point_mask = valid_point_mask & (lidar_points_uv[:, 0] >= 0) \
                    & (lidar_points_uv[:, 0] < W - 1) \
                    & (lidar_points_uv[:, 1] >= 0) \
                    & (lidar_points_uv[:, 1] < H - 1)   # Left shape: (num_points,)
                valid_lidar_points_cam = lidar_points_cam[valid_point_mask][:, :, 0]    # Left shape: (num_points, 3)
                valid_lidar_points_uv = lidar_points_uv[valid_point_mask]  # Left shape: (num_points, 2)
                valid_lidar_points_d = lidar_points_d[valid_point_mask]   # Left shape: (num_points, 1)

                # Save lidar points
                save_lidar_path = os.path.join(mmOmni3d_lidar_path, lidar_path.rsplit('/')[-1])
                valid_lidar_points_cam.astype(np.float32).tofile(save_lidar_path)
                
                # Add lidar point path to json
                lidar_json_path = os.path.join('nuScenes/samples/LIDAR_TOP', lidar_path.rsplit('/')[-1]) 
                omni3d_nus_json['images'][cnt]['depth_file_path'] = lidar_json_path

        json_save_path = os.path.join(mmOmni3d_json_path, omni3d_nus_json_name)
        with open(json_save_path, 'w') as write_f:
            json.dump(omni3d_nus_json, write_f)

    print('Done!')

if __name__ == '__main__':
    nuscenes_root = '...'  # Change it to the root folder path of the downloaded nuScenes dataset.
    omni3d_root = '...'  # Change it to the root folder path of saving the MM-Omni3D dataset.
    prepare_mm_omni3d_nuscenes(nus_root = nuscenes_root, omni3d_root = omni3d_root)