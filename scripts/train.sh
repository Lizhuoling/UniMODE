exp_id=GLIP_demo

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  OUTPUT_DIR output/evaluation \
  #MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \