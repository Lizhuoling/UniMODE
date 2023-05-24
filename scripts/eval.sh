exp_id=Debug

CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
  --eval-only \
  --config-file configs/$exp_id.yaml \
  OUTPUT_DIR output/$exp_id \
  MODEL.WEIGHTS output/OV_base1_kittitrain_noGLIP_closehead_detr/model_final.pth
  #MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \