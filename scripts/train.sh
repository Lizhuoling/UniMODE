exp_id=OV_base1_kittitrain_noGLIP

CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 1 \
  OUTPUT_DIR output/$exp_id \