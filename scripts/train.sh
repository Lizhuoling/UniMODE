exp_id=BEVglobal_base4_omni3d_out_imgpad2

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #MODEL.WEIGHTS_PRETRAIN output/BEVglobal_base3_omni3d_out_intrinsic/model_recent.pth
  #--resume \
