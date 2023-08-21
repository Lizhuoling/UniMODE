exp_id=TwoStage_base3_omni3d_convenc

#ulimit -n 65536

python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url tcp://127.0.0.1:12345 \
  --resume \
  OUTPUT_DIR output/$exp_id \
  MODEL.WEIGHTS_PRETRAIN output/$exp_id/model_recent.pth \
  #--dist-url tcp://ji-jupyter-6937317828539244544-master-0.ji-jupyter-6937317828539244544:12345 \

  #--dist-url tcp://127.0.0.1:12345 \
