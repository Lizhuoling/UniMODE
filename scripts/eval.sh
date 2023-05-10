exp_id=GLIP_demo

CUDA_VISIBLE_DEVICES=2 python tools/train_net.py \
  --eval-only \
  --config-file configs/$exp_id.yaml \
  OUTPUT_DIR output/evaluation \
  #MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \