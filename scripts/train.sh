exp_id=bev_convnext_deformable_mutual_outdoor

#ulimit -n 65536

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py \
  --config-file configs/$exp_id.yaml \
  --num-gpus 4 \
  --num-machines 1 \
  --machine-rank 0 \
  --dist-url tcp://127.0.0.1:12345 \
  OUTPUT_DIR output/$exp_id \
  #--resume \
  #MODEL.WEIGHTS_PRETRAIN output/$exp_id/model_recent.pth \

  #--dist-url tcp://ji-jupyter-6937317828539244544-master-0.ji-jupyter-6937317828539244544:12345 \
  #--dist-url tcp://127.0.0.1:12345 \
