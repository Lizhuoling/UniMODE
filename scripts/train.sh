exp_id=OV_base2_nus_vov_reso640

CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 2 \
  --dist-url tcp://127.0.0.1:12344 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/OV_base2_omni3d_out_vov_visualfuse_loadpretrained/model_recent.pth