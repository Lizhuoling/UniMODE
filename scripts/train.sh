exp_id=OV_base2_omni3d_out_vov

CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 2 \
  OUTPUT_DIR output/$exp_id \
  #MODEL.WEIGHTS_PRETRAIN output/OV_base1_omni3d_out_noGLIP_closehead_vov/model_recent.pth