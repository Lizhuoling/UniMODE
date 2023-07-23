exp_id=Base_Omni3D_demo

CUDA_VISIBLE_DEVICES=1 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 1 \
  --dist-url tcp://127.0.0.1:12346 \
  OUTPUT_DIR output/$exp_id \
  #MODEL.WEIGHTS_PRETRAIN output/BEVglobal_base3_omni3d_out_intrinsic/model_recent.pth
  #--resume \
