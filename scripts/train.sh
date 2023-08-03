exp_id=TwoStage_base2_sun_range_loc3x

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #MODEL.WEIGHTS_PRETRAIN output/TwoStage_base2_sun_range/model_recent.pth \
  #--resume \
