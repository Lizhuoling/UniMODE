exp_id=mm_nusar_convnext

CUDA_VISIBLE_DEVICES=3 python tools/train_net.py \
  --eval-only \
  --config-file configs/$exp_id.yaml \
  OUTPUT_DIR output/$exp_id \
  MODEL.WEIGHTS output/$exp_id/model_recent.pth
  #MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth