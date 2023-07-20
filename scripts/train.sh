exp_id=BEVglobal_base3_sunrgbd2

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 1 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/OV_base2_omni3d_out_vov_visualfuse_loadpretrained/model_recent.pth
