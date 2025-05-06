import pdb
import os
import sys
import tqdm
import json
import requests
import struct
import numpy as np
import matplotlib.pyplot as plt

from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol

def prepare_mm_omni3d_objectron(omni3d_root, objectron_url, objectron_meta_root):
    omni3d_json_path = os.path.join(omni3d_root, 'Omni3D')
    mmOmni3d_json_path = os.path.join(omni3d_root, 'MM-Omni3D')
    mmOmni3d_lidar_path = os.path.join(omni3d_root, 'objectron/pointcloud')

    omni3d_objectron_json_list = ['Objectron_train.json', 'Objectron_val.json', 'Objectron_test.json']
    for omni3d_objectron_json_name in omni3d_objectron_json_list:
        omni3d_objectron_json_file_path = os.path.join(omni3d_json_path, omni3d_objectron_json_name)
        with open(omni3d_objectron_json_file_path, 'r') as read_f:
            omni3d_objectron_json = json.load(read_f)
            omni3d_objectron_images = omni3d_objectron_json['images']
            print("Process {}...".format(omni3d_objectron_json_name))

            for cnt, objectron_image_dict in enumerate(tqdm.tqdm(omni3d_objectron_images)):

                img_file_name = objectron_image_dict['file_path'].rsplit('/')[-1]
                data_id = img_file_name.split('.')[0]
                category, _, batch_id, video_id, frame_id = data_id.rsplit('_', 4)
                metadata_id = category + '_' + batch_id + '_' + video_id
                metadata_write_path = os.path.join(objectron_meta_root, metadata_id + '.pbdata')
                save_lidar_path = os.path.join(mmOmni3d_lidar_path, data_id + '.pcd.bin')
                
                sequence_geometry = get_geometry_data(metadata_write_path)
                transform, intrinsics, projection, view, scene_points_3d = sequence_geometry[int(frame_id)]

                img_file_path = os.path.join(omni3d_root, objectron_image_dict['file_path'])
                img = plt.imread(img_file_path)
                height, width, _ = img.shape

                if scene_points_3d.shape[0] != 0:
                    scene_points_2d, scene_points_depth = project_points(scene_points_3d, projection, view, width, height) 
                    valid_mask = (scene_points_2d[:, 0] >= 0) & (scene_points_2d[:, 0] <= width - 1) & (scene_points_2d[:, 1] >= 0) & (scene_points_2d[:, 1] <= height - 1) & (scene_points_depth > 0)
                    scene_points_2d = scene_points_2d[valid_mask]
                    scene_points_depth = scene_points_depth[valid_mask]
                    points_uvd = np.concatenate((scene_points_2d * scene_points_depth[:, None], scene_points_depth[:, None]), axis = -1)
                    K = np.array(objectron_image_dict['K'])
                    points_3d = (np.linalg.inv(K) @ points_uvd.T).T # Left shape: (num_points, 3)
                else:
                    points_3d = scene_points_3d

                # Save lidar points
                points_3d.astype(np.float32).tofile(save_lidar_path)
                    
                # Add lidar point path to json
                lidar_json_path = os.path.join('objectron/pointcloud', data_id + '.pcd.bin') 
                omni3d_objectron_json['images'][cnt]['depth_file_path'] = lidar_json_path
                
        json_save_path = os.path.join(mmOmni3d_json_path, omni3d_objectron_json_name)
        with open(json_save_path, 'w') as write_f:
            json.dump(omni3d_objectron_json, write_f)
        print('Done!')
                

def project_points(points, projection_matrix, view_matrix, width, height):
    p_3d = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1).T
    p_3d_cam = np.matmul(view_matrix, p_3d)
    p_2d_proj = np.matmul(projection_matrix, p_3d_cam)

    # Project the points
    p_2d_ndc = p_2d_proj[:-1, :] / p_2d_proj[-1, :]
    p_2d_ndc = p_2d_ndc.T

    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    x = p_2d_ndc[:, 1]
    y = p_2d_ndc[:, 0]
    pixels = np.copy(p_2d_ndc)
    pixels[:, 0] = ((1 + x) * 0.5) * width
    pixels[:, 1] = ((1 + y) * 0.5) * height   
    pixels = pixels.astype(int)
    return pixels[:, :2], p_2d_proj[-1, :]          

def get_geometry_data(geometry_filename):
    sequence_geometry = []
    with open(geometry_filename, 'rb') as pb:
        proto_buf = pb.read()

        i = 0
        frame_number = 0

        while i < len(proto_buf):
            # Read the first four Bytes in little endian '<' integers 'I' format
            # indicating the length of the current message.
            msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
            i += 4
            message_buf = proto_buf[i:i + msg_len]
            i += msg_len
            frame_data = ar_metadata_protocol.ARFrame()
            frame_data.ParseFromString(message_buf)


            transform = np.reshape(frame_data.camera.transform, (4, 4))
            projection = np.reshape(frame_data.camera.projection_matrix , (4, 4))
            view = np.reshape(frame_data.camera.view_matrix , (4, 4))
            intrinsics = np.reshape(frame_data.camera.intrinsics, (3, 3))
            position = transform[:3, -1]

            current_points = [np.array([v.x, v.y, v.z])
                              for v in frame_data.raw_feature_points.point]
            current_points = np.array(current_points)
            
            sequence_geometry.append((transform, intrinsics, projection, view, current_points))
    return sequence_geometry

if __name__ == '__main__':
    objectron_meta_root = '...' # Change it to the root folder path of saving the ObjectronMetaData.
    omni3d_root = '...'  # Change it to the root folder path of saving the MM-Omni3D dataset.
    prepare_mm_omni3d_objectron(
        omni3d_root = omni3d_root,
        objectron_url = 'https://storage.googleapis.com/objectron',
        objectron_meta_root = objectron_meta_root,
    )