conda create --name omni3d -y python=3.8

pip install torch-1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

pip install openmim
mim install mmcv-full==1.4.2
pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2
rm -rf mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.18.1
pip install -e .
cd ..

pip install cython opencv-python timm transformers einops ftfy
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
#python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

git clone https://github.com/facebookresearch/pytorch3d.git # version 0.7.4
cd pytorch3d 
pip install -e .
cd ..

python setup_maskrcnn_benchmark.py