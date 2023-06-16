conda create --name omni3d -y python=3.8

#pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
#pip install -r requirements.txt

pip install openmim
#pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
#mim install mmdet   # 3.0.0
#pip install "mmsegmentation>=1.0.0"
#mim install "mmdet3d>=1.1.0rc0"
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
pip install mmdet==v2.28.2  
pip install mmsegmentation==0.30.0
pip install mmdet3d==v1.0.0rc4

pip install cython opencv-python timm transformers einops ftfy opencv-python-headless
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

apt-get autoremove nvidia-cuda-toolkit
# Addthe following to .bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=/usr/local/cuda

rm -rf detectron2
git clone https://gitee.com/twilightLZL/detectron2.git
python -m pip install -e detectron2

rm -rf pytorch3d
git clone https://gitee.com/twilightLZL/pytorch3d.git
cd pytorch3d 
rm -rf build
pip install -e .
cd ..

rm -rf build
python setup_maskrcnn_benchmark.py develop

pip install setuptools==59.5.0