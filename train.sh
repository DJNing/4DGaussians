export CUDA_VISIBLE_DEVICES=2
python train.py -s data/dnerf/laptop_10211 --port 6017 --expname "dnerf/laptop_10211" --configs arguments/dnerf/bouncingballs.py 