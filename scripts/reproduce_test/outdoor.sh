#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/megadepth_test_1500.py"
main_cfg_path="configs/edm/outdoor/edm_base.py"

dump_dir="dump/edm_outdoor"
profiler_name="inference"
n_nodes=1
n_gpus_per_node=1
torch_num_workers=8
batch_size=1

ckpt_path="/workspace/EDM/logs/tb_logs/edm_outdoor/version_6/checkpoints/epoch=29-auc@5=0.551-auc@10=0.710-auc@20=0.826.ckpt"
size=1152 # follow ELoFTR's setting

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --gpus=${n_gpus_per_node} \
    --num_nodes=${n_nodes} \
    --accelerator="ddp" \
    --batch_size=${batch_size} \
    --num_workers=${torch_num_workers} \
    --ckpt_path=${ckpt_path} \
    --dump_dir ${dump_dir} \
    --deter \
    --W ${size} \
    --H ${size} \

