exp_id=debug

#ulimit -n 1024000

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 1 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/$exp_id/model_recent.pth \

  #TwoStage_base2_nusar_sparse1e-3
  #TwoStage_base2_nusar_sparse1e-3_finegrid
  #TwoStage_base2_nusar_sparse1e-3_finegriddepth
  # center head decouple focal length
  # center head increase receptive field
