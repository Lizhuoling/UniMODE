exp_id=TwoStage_base2_nusar_voxelfeat_h1

#ulimit -n 65536

python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 2 \
  --num-machines 2 \
  --machine-rank 0 \
  --dist-url tcp://ji-jupyter-7115764936211730432-master-0.ji-jupyter-7115764936211730432:12345 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/$exp_id/model_recent.pth \

  #--dist-url tcp://127.0.0.1:12345 \
