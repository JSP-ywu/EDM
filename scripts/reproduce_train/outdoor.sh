#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=832

data_cfg_path="configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/edm/outdoor/edm_base.py"

n_nodes=1
n_gpus_per_node=4
torch_num_workers=8
batch_size=4
pin_memory=true
exp_name="edm_"
ckpt=""
resume=False
ew=0.2
et=1.0
# --- parse only ew / et from CLI ---
# usage:
#   bash outdoor.sh --ew 0.1 --et 1.5
#   bash outdoor.sh --ew=0.1 --et=1.5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ew=*) ew="${1#*=}"; shift ;;
    --ew)   ew="$2"; shift 2 ;;
    --et=*) et="${1#*=}"; shift ;;
    --et)   et="$2"; shift 2 ;;
    --exp_name=*) exp_name="${1#*=}"; shift ;;
    --exp_name)   exp_name="$2"; shift 2 ;;
    *)      shift ;;  # ignore other args
  esac
done

echo "[outdoor.sh] exp_name=${exp_name} ew=${ew:-(default)} et=${et:-(default)}"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} \
    --num_nodes=${n_nodes} \
    --batch_size=${batch_size} \
    --num_workers=${torch_num_workers} \
    --pin_memory=${pin_memory} \
    --ckpt_path=${ckpt} \
    --resume=${resume}\
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=500 \
    --num_sanity_val_steps=10 \
    --benchmark=true \
    --max_epochs=30 \
    --split_data_idx=1 \
    --ew=${ew} \
    --et=${et}

