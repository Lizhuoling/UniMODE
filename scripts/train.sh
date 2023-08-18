exp_id=TwoStage_base2_nusar_voxelfeat

#ulimit -n 1024000

python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  --num_machines 2 \
  --machine_rank 0 \
  --dist_url ji-jupyter-7115764936211730432-master-0.ji-jupyter-7115764936211730432 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/$exp_id/model_recent.pth \

  #TwoStage_base2_nusar_sparse1e-3
  #TwoStage_base2_nusar_sparse1e-3_finegrid
  #TwoStage_base2_nusar_sparse1e-3_finegriddepth
  # center head decouple focal length
  # center head increase receptive field
