exp_id=OV_base1_kittitrain

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  OUTPUT_DIR output/$exp_id \