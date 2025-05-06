## KITTI

Step 1: Download the KITTI 3D object detection dataset from the [official website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

Step 2: Change the path variables (kitti_root and omni3d_root) in kitti_prepare.py and run it to prepare the KITTI subset in MM-Omni3D.

## nuScenes

Step 1: Download the nuScenes dataset from the [official website](https://www.nuscenes.org/nuscenes).

Step 2: Change the path variables (nuscenes_root and omni3d_root) in nuscenes_prepare.py and run it to prepare the nuScenes subset in MM-Omni3D.

## Sun RGB-D

Step 1: Download the Sun RGB-D dataset from the [official website](https://rgbd.cs.princeton.edu/).

Step 2: Change the path variables (sunrgbd_root and omni3d_root) in sunrgbd_prepare.py and run it to prepare the Sun RGB-D subset in MM-Omni3D.

## ARKitScenes

Step 1: Clone the ARKitScenes official repository from [here](https://github.com/apple/ARKitScenes.git).

Step 2: Run the following command to download the original dataset:
```
python3 ARKitScenes/download_data.py raw --video_id_csv ARKitScenes/raw/raw_train_val_splits.csv --download_dir $ARKitScenes_save_root --raw_dataset_assets lowres_depth
```
where $ARKitScenes_save_root is the saving path.

Step 3: Change the path variables (ar_root and omni3d_root) in arkitscenes_prepare.py and run it to prepare the ARKitScenes subset in MM-Omni3D.

## Hypersim

Step 1: Download the Hypersim dataset following the guide in [Omni3D](https://github.com/facebookresearch/omni3d).

Step 2: Change the path variables (hypersim_root, hypersim_meta_root, omni3d_root, and MM_omni3d_root) in hypersim_prepare.py and run it to prepare the Hypersim subset in MM-Omni3D.

## Objectron

Step 1: Download the Objectron dataset following the guide in [Omni3D](https://github.com/facebookresearch/omni3d).

Step 2: Change the path variables (objectron_meta_root, omni3d_root) in objectron_prepare.py and run it to prepare the Objectron subset in MM-Omni3D.