import pdb
import os
import tqdm
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import camtools as ct   # For ct.convert.T_blender_to_pinhole, use version 0.1.1. For T_opengl_to_opencv, use version 0.1.4.

def prepare_mm_omni3d_hypersim(hypersim_root, hypersim_meta_root, omni3d_root, MM_omni3d_root):
    omni3d_json_path = os.path.join(omni3d_root, 'Omni3D')
    mmOmni3d_json_path = os.path.join(MM_omni3d_root, 'MM-Omni3D')
    mmOmni3d_lidar_path = os.path.join(MM_omni3d_root, 'hypersim/pointcloud')

    hypersim_meta = pd.read_csv(hypersim_meta_root)

    # Process lidar point and save it.
    omni3d_hypersim_json_list = ['Hypersim_train.json', 'Hypersim_val.json', 'Hypersim_test.json']
    #omni3d_hypersim_json_list = ['Hypersim_val.json',]
    for omni3d_hypersim_json_name in omni3d_hypersim_json_list:
        omni3d_hypersim_json_file_path = os.path.join(omni3d_json_path, omni3d_hypersim_json_name)
        with open(omni3d_hypersim_json_file_path, 'r') as read_f:
            omni3d_hypersim_json = json.load(read_f)
            omni3d_hypersim_images = omni3d_hypersim_json['images']
            print("Process {}...".format(omni3d_hypersim_json_name))
            for cnt, omni3d_hypersim_image_dict in enumerate(tqdm.tqdm(omni3d_hypersim_images)):

                _, scene_name, _, cam_id, img_file_name = omni3d_hypersim_image_dict['file_path'].rsplit('/', 4)
                cam_name = 'cam_' + cam_id.split('_')[2]
                data_id = "{}_{}_frame{}".format(scene_name, cam_name, img_file_name.split('.')[1])
                omni3d_K = np.array(omni3d_hypersim_image_dict['K'])
                frame_id = int(img_file_name.split('.')[1])

                img_file_path = os.path.join(omni3d_root, omni3d_hypersim_image_dict['file_path'])
                img = plt.imread(img_file_path) # img shape: (img_h, img_w, 3)
                img_h, img_w, _ = img.shape
                
                data_meta = hypersim_meta.loc[hypersim_meta['scene_name'] == scene_name]
                width_pixels = int(data_meta["settings_output_img_width"])
                height_pixels = int(data_meta["settings_output_img_height"])
                M_proj = np.array([
                    [data_meta['M_proj_00'].item(), data_meta['M_proj_01'].item(), data_meta['M_proj_02'].item(), data_meta['M_proj_03'].item()],
                    [data_meta['M_proj_10'].item(), data_meta['M_proj_11'].item(), data_meta['M_proj_12'].item(), data_meta['M_proj_13'].item()],
                    [data_meta['M_proj_20'].item(), data_meta['M_proj_21'].item(), data_meta['M_proj_22'].item(), data_meta['M_proj_23'].item()],
                    [data_meta['M_proj_30'].item(), data_meta['M_proj_31'].item(), data_meta['M_proj_32'].item(), data_meta['M_proj_33'].item()],
                ])

                camera_dir = os.path.join(hypersim_root, scene_name, "_detail", cam_name)
                camera_positions_hdf5_file = os.path.join(camera_dir, "camera_keyframe_positions.hdf5")
                camera_orientations_hdf5_file = os.path.join(camera_dir, "camera_keyframe_orientations.hdf5")
                
                geometry_pos_path = os.path.join(hypersim_root, scene_name, 'images', cam_id.replace('final_preview', 'geometry_hdf5'), img_file_name.replace('tonemap.jpg', 'position.hdf5'))
                with h5py.File(geometry_pos_path, "r") as f:
                    geo_pos = f["dataset"][:]   # geo_pos shape: (img_h, img_w, 3)
                voxel_scale_path = os.path.join(hypersim_root, scene_name, '_detail/metadata_scene.csv')
                voxel_scale = pd.read_csv(voxel_scale_path).loc[0, 'parameter_value']
                #geo_pos = geo_pos * voxel_scale # Left shape: (img_h, img_w, 3) # For debug

                with h5py.File(camera_positions_hdf5_file, "r") as f:
                    camera_positions = f["dataset"][:]
                with h5py.File(camera_orientations_hdf5_file, "r") as f:
                    camera_orientations = f["dataset"][:]
                M_screen_from_ndc = np.array([
                    [0.5 * (width_pixels - 1), 0, 0, 0.5 * (width_pixels - 1)],
                    [0, -0.5 * (height_pixels - 1), 0, 0.5 * (height_pixels - 1)],
                    [0, 0, 0.5, 0.5],
                    [0, 0, 0, 1.0],
                ])
                camera_position_world = camera_positions[frame_id]
                R_world_from_cam = camera_orientations[frame_id]
                t_world_from_cam = np.array(camera_position_world).T
                R_cam_from_world = np.array(R_world_from_cam).T
                t_cam_from_world = -R_cam_from_world @ t_world_from_cam
                M_cam_from_world = np.eye(4)
                M_cam_from_world[:3, :3] = R_cam_from_world
                M_cam_from_world[:3, 3] = t_cam_from_world

                T = ct.convert.T_blender_to_pinhole(M_cam_from_world)
                K_opengl = M_screen_from_ndc @ M_proj
                fx = K_opengl[0, 0]
                fy = -K_opengl[1, 1]
                cx = -K_opengl[0, 2]
                cy = -K_opengl[1, 2]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                H_geo_pos = np.concatenate((geo_pos, np.ones((img_h, img_w, 1), dtype=geo_pos.dtype)), axis=-1)
                H_geo_pos = H_geo_pos.reshape(-1, 4)
                H_geo_pos_cam = (T @ H_geo_pos.T).T[:, :3]  # The 3D points in 3D camera coordinate system. 

                geo_pos_uvd = (K @ H_geo_pos_cam.T).T
                depth_valid_mask = geo_pos_uvd[..., 2] > 0
                geo_pos_uvd = geo_pos_uvd[depth_valid_mask]
                geo_pos_uv = geo_pos_uvd[:, :2] / geo_pos_uvd[:, 2:]
                geo_pos_d = geo_pos_uvd[:, 2:]
                uv_img_mask = (geo_pos_uv[:, 0] >= 0) & (geo_pos_uv[:, 0] <= img_w - 1) & (geo_pos_uv[:, 1] >= 0) & (geo_pos_uv[:, 1] <= img_h - 1)
                geo_pos_uv = geo_pos_uv[uv_img_mask]
                geo_pos_d = geo_pos_d[uv_img_mask]

                # This block of code is controversial.
                update_geo_pos_d = geo_pos_d * voxel_scale
                geo_pos_uvd = geo_pos_uv * update_geo_pos_d
                point_xyz = (np.linalg.inv(K) @ np.concatenate((geo_pos_uvd, update_geo_pos_d), axis = -1).T).T

                save_lidar_path = os.path.join(mmOmni3d_lidar_path, data_id + '.pcd.bin')
                point_xyz.astype(np.float32).tofile(save_lidar_path)

                lidar_json_path = os.path.join('hypersim/pointcloud', data_id + '.pcd.bin') 
                omni3d_hypersim_json['images'][cnt]['depth_file_path'] = lidar_json_path

        json_save_path = os.path.join(mmOmni3d_json_path, omni3d_hypersim_json_name)
        with open(json_save_path, 'w') as write_f:
            json.dump(omni3d_hypersim_json, write_f)
        print('Done!')

def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth

def prepare_mm_omni3d_hypersim2(hypersim_root, hypersim_meta_root, omni3d_root, MM_omni3d_root):
    omni3d_json_path = os.path.join(omni3d_root, 'Omni3D')
    mmOmni3d_json_path = os.path.join(MM_omni3d_root, 'MM-Omni3D')
    mmOmni3d_lidar_path = os.path.join(MM_omni3d_root, 'hypersim/pointcloud')

    hypersim_meta = pd.read_csv(hypersim_meta_root)

    # Process lidar point and save it.
    omni3d_hypersim_json_list = ['Hypersim_train.json', 'Hypersim_val.json', 'Hypersim_test.json']
    for omni3d_hypersim_json_name in omni3d_hypersim_json_list:
        omni3d_hypersim_json_file_path = os.path.join(omni3d_json_path, omni3d_hypersim_json_name)
        with open(omni3d_hypersim_json_file_path, 'r') as read_f:
            omni3d_hypersim_json = json.load(read_f)
            omni3d_hypersim_images = omni3d_hypersim_json['images']
            print("Process {}...".format(omni3d_hypersim_json_name))
            for cnt, omni3d_hypersim_image_dict in enumerate(tqdm.tqdm(omni3d_hypersim_images)):

                _, scene_name, _, cam_id, img_file_name = omni3d_hypersim_image_dict['file_path'].rsplit('/', 4)
                cam_name = 'cam_' + cam_id.split('_')[2]
                data_id = "{}_{}_frame{}".format(scene_name, cam_name, img_file_name.split('.')[1])
                omni3d_K = np.array(omni3d_hypersim_image_dict['K'])
                frame_id = int(img_file_name.split('.')[1])

                img_file_path = os.path.join(omni3d_root, omni3d_hypersim_image_dict['file_path'])
                img = plt.imread(img_file_path) # img shape: (img_h, img_w, 3)

                depth_path = os.path.join(hypersim_root, scene_name, 'images', "scene_{}_geometry_hdf5".format(cam_name), "frame.{}.depth_meters.hdf5".format(img_file_name.split('.')[1]))
                depth_fd = h5py.File(depth_path, "r")
                distance_meters = np.array(depth_fd['dataset'])
                depth = hypersim_distance_to_depth(distance_meters)
                
                data_meta = hypersim_meta.loc[hypersim_meta['scene_name'] == scene_name]
                width_pixels = int(data_meta["settings_output_img_width"])
                height_pixels = int(data_meta["settings_output_img_height"])
                M_proj = np.array([
                    [data_meta['M_proj_00'].item(), data_meta['M_proj_01'].item(), data_meta['M_proj_02'].item(), data_meta['M_proj_03'].item()],
                    [data_meta['M_proj_10'].item(), data_meta['M_proj_11'].item(), data_meta['M_proj_12'].item(), data_meta['M_proj_13'].item()],
                    [data_meta['M_proj_20'].item(), data_meta['M_proj_21'].item(), data_meta['M_proj_22'].item(), data_meta['M_proj_23'].item()],
                    [data_meta['M_proj_30'].item(), data_meta['M_proj_31'].item(), data_meta['M_proj_32'].item(), data_meta['M_proj_33'].item()],
                ])
                camera_dir = os.path.join(hypersim_root, scene_name, "_detail", cam_name)
                camera_positions_hdf5_file = os.path.join(camera_dir, "camera_keyframe_positions.hdf5")
                camera_orientations_hdf5_file = os.path.join(camera_dir, "camera_keyframe_orientations.hdf5")
                with h5py.File(camera_positions_hdf5_file, "r") as f:
                    camera_positions = f["dataset"][:]
                with h5py.File(camera_orientations_hdf5_file, "r") as f:
                    camera_orientations = f["dataset"][:]
                M_screen_from_ndc = np.array([
                    [0.5 * (width_pixels - 1), 0, 0, 0.5 * (width_pixels - 1)],
                    [0, -0.5 * (height_pixels - 1), 0, 0.5 * (height_pixels - 1)],
                    [0, 0, 0.5, 0.5],
                    [0, 0, 0, 1.0],
                ])
                K_opengl = M_screen_from_ndc @ M_proj
                fx = K_opengl[0, 0]
                fy = -K_opengl[1, 1]
                cx = -K_opengl[0, 2]
                cy = -K_opengl[1, 2]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                img_h, img_w, _ = img.shape
                assert img_h == depth.shape[0] and img_w == depth.shape[1]
                h_coor = np.linspace(0, img_h-1, img_h)
                w_coor = np.linspace(0, img_w-1, img_w)
                w_coor, h_coor = np.meshgrid(w_coor, h_coor)
                geo_pos_uv = np.concatenate((w_coor[..., None], h_coor[..., None]), axis = -1).reshape(img_h * img_w, 2)
                geo_pos_d = depth[..., None].reshape(img_h * img_w, 1)
                geo_pos_uvd = geo_pos_uv * geo_pos_d
                point_xyz = (np.linalg.inv(K) @ np.concatenate((geo_pos_uvd, geo_pos_d), axis = -1).T).T

                save_lidar_path = os.path.join(mmOmni3d_lidar_path, data_id + '.pcd.bin')
                point_xyz.astype(np.float32).tofile(save_lidar_path)
                lidar_json_path = os.path.join('hypersim/pointcloud', data_id + '.pcd.bin') 
                omni3d_hypersim_json['images'][cnt]['depth_file_path'] = lidar_json_path

        json_save_path = os.path.join(mmOmni3d_json_path, omni3d_hypersim_json_name)
        with open(json_save_path, 'w') as write_f:
            json.dump(omni3d_hypersim_json, write_f)
        print('Done!')

if __name__ == '__main__':
    prepare_mm_omni3d_hypersim2(
        hypersim_root = '...',   # The path to Hypersim data.
        hypersim_meta_root = '...', # The path to ml-hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv
        omni3d_root = '/hszhao-f1/h3011051/data/Omni3D',    # The path to the path of saving the original Omni3D dataset.
        MM_omni3d_root = '...', # The path for saving MM-Omni3D.
    )