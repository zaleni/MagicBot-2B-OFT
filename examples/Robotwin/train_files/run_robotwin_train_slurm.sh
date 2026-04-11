#!/bin/bash
#SBATCH --job-name=robotwin_oft3d
#SBATCH -p h100
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Optional NCCL NIC/HCA pinning. Set these in your shell if the cluster needs them.
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export WANDB_MODE=offline

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

###########################################################################################
# Only edit the paths / name below if needed.
Framework_name=QwenOFT3D
base_vlm=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Qwen3.5-2B
da3_model_path=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/DA3-LARGE-1-1
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
run_root_dir=./results/Checkpoints
data_root=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robotwin_all_50
run_id=robotwin_selected_50_future3d_qwen35_2b_oft3d_n2g16
###########################################################################################

data_mix=robotwin_selected_50_future3d
freeze_module_list=''
per_device_batch_size=8

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "Cannot find conda.sh. Please update the conda path in this script."
  exit 1
fi

conda activate starVLA

export GPUS_PER_NODE=8
export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + RANDOM % 10000))

echo "SLURM_NNODES=${SLURM_NNODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "TOTAL_GPUS=${TOTAL_GPUS}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
set -euo pipefail
cd "'"${REPO_ROOT}"'"
echo "Host=$(hostname)  SLURM_PROCID=$SLURM_PROCID"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip "'"${MASTER_ADDR}"'" \
  --main_process_port "'"${MASTER_PORT}"'" \
  --machine_rank "$SLURM_PROCID" \
  --num_machines "'"${SLURM_NNODES}"'" \
  --num_processes "'"${TOTAL_GPUS}"'" \
  starVLA/training/train_starvla.py \
  --config_yaml "'"${config_yaml}"'" \
  --framework.name "'"${Framework_name}"'" \
  --framework.qwenvl.base_vlm "'"${base_vlm}"'" \
  --framework.future3d.da3_model_path_or_name "'"${da3_model_path}"'" \
  --datasets.vla_data.per_device_batch_size "'"${per_device_batch_size}"'" \
  --datasets.vla_data.data_root_dir "'"${data_root}"'" \
  --datasets.vla_data.data_mix "'"${data_mix}"'" \
  --trainer.freeze_modules "'"${freeze_module_list}"'" \
  --trainer.max_train_steps 90000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 25 \
  --trainer.eval_interval 1000 \
  --run_root_dir "'"${run_root_dir}"'" \
  --run_id "'"${run_id}"'" \
  --wandb_project MagicBot-2B-OFT3D \
  --wandb_entity zaleni
'
