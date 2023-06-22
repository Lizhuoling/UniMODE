exp_id=OV_base1_omni3d_out_noGLIP_closehead_vov_debug

python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 1 \
  OUTPUT_DIR output/$exp_id \
  #MODEL.WEIGHTS_PRETRAIN output/OV_base1_kittitrain_noGLIP_closehead_detr/model_recent.pth