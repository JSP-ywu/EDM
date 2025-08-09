#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=832

data_cfg_path="configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/edm/outdoor/edm_base.py"

n_nodes=1
n_gpus_per_node=8
torch_num_workers=8
batch_size=4
pin_memory=true
exp_name="depth_fusion_depth_map_diffbb_f16_da_map"
ckpt=""
pre_extracted_depth=False

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} \
    --num_nodes=${n_nodes} \
    --accelerator="ddp" \
    --batch_size=${batch_size} \
    --num_workers=${torch_num_workers} \
    --pin_memory=${pin_memory} \
    --ckpt_path=${ckpt} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=500 \
    --num_sanity_val_steps=10 \
    --benchmark=true \
    --max_epochs=30 \
    --split_data_idx=1 \
    --pre_extracted_depth=${pre_extracted_depth}\
