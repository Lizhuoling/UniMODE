exp_id=TwoStage_base2_nusar_voxelfeat

#ulimit -n 1024000

python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  --num-machines 2 \
  --machine-rank 0 \
  --dist-url ji-jupyter-7115764936211730432-master-0.ji-jupyter-7115764936211730432 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/$exp_id/model_recent.pth \

  #--dist-url tcp://127.0.0.1:12345 \
