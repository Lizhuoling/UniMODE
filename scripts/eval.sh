exp_id=OV_base1_omni3d_out_noGLIP_closehead_vov_debug

CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
  --eval-only \
  --config-file configs/$exp_id.yaml \
  OUTPUT_DIR output/$exp_id \
  MODEL.WEIGHTS output/OV_base1_omni3d_out_noGLIP_closehead_vov/model_recent.pth
  #MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth