exp_id=BEVglobal_base3_omni3d_out_glip

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/OV_base2_omni3d_out_vov_visualfuse_loadpretrained/model_recent.pth
