exp_id=OV_base1_kittitrain_noGLIP_closehead_detr

python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  OUTPUT_DIR output/$exp_id \
  #MODEL.WEIGHTS output/OV_base1_kittitrain_noGLIP_closehead_detr/model_recent.pth \