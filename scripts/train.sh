exp_id=OV_base1_omni3d_out_vov

python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  OUTPUT_DIR output/$exp_id \
  #MODEL.WEIGHTS_PRETRAIN output/OV_base1_omni3d_out_noGLIP_closehead_vov/model_recent.pth