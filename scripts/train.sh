exp_id=TwoStage_base2_nusar_sparse1e-3

ulimit -n 1024000

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #MODEL.WEIGHTS_PRETRAIN output/$exp_id/model_recent.pth \
  #--resume \

  #TwoStage_base2_nusar
  #TwoStage_base2_nusar_newlss
  #TwoStage_base2_nusar_gtcenter
  #TwoStage_base2_nusar_sparse1e-3
  #TwoStage_base2_nusar_sparse1e-3_finegrid
  #TwoStage_base2_nusar_sparse1e-3_finegriddepth
