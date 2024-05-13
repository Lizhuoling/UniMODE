exp_id=debug

#ulimit -n 65536

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 1 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/$exp_id/model_recent.pth \
