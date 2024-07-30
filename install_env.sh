conda create --name UniMODE -y python=3.8
conda activate UniMODE

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install openmim
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install mmdet==v2.28.2  
pip install mmsegmentation==0.30.0
pip install mmdet3d==v1.0.0rc4

pip install cython opencv-python timm transformers einops ftfy opencv-python-headless tensorboardX tensorflow rope ninja
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d 
pip install -e .
cd ..

python setup_voxel_pooling.py develop

pip install setuptools==59.5.0
pip install Pillow==8.2.0
pip install numba==0.56.4

cd model/deformable_ops
python setup.py develop
cd ../..