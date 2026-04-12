#!/bin/bash
#SBATCH --job-name=robotwin_oft3d
#SBATCH -p h100x
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Optional NCCL NIC/HCA pinning. Set these in your shell if the cluster needs them.
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA

export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1200
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_DIST_TIMEOUT_MINUTES=240
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export WANDB_MODE=offline
# export HF_ENDPOINT=https://hf-mirror.com
cd /HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/MagicBot-2B-OFT
echo "Current working directory: $(pwd)"

export CUDA_HOME=/APP/u22/ai_x86/CUDA/12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH=$CUDA_HOME/bin:$PATH
###########################################################################################
# Only edit the paths / name below if needed.
Framework_name=QwenOFT3D
base_vlm=/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/Qwen3.5-2B
da3_model_path=/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/DA3-LARGE-1.1
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
run_root_dir=/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/MagicBot-2B-OFT/results/Checkpoints
data_root=/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/data/RoboTwin-Randomized
run_id=robotwin_selected_50_future3d_qwen35_2b_oft3d_32_q216_80k_2
attn_implementation=flash_attention_2
future3d_num_query_tokens=216
accelerate_config=${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2_stable.yaml}
###########################################################################################

data_mix=robotwin_selected_50_future3d
freeze_module_list=${FREEZE_MODULE_LIST:-''}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-4}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

if [ -f /HOME/uestc_jksong/uestc_jksong_1/miniconda3/etc/profile.d/conda.sh ]; then
  source /HOME/uestc_jksong/uestc_jksong_1/miniconda3/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
else
  echo "Cannot find conda.sh. Please update the conda path in this script."
  exit 1
fi

conda activate magicbot

export GPUS_PER_NODE=8
export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + RANDOM % 10000))

echo "SLURM_NNODES=${SLURM_NNODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "TOTAL_GPUS=${TOTAL_GPUS}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "attn_implementation=${attn_implementation}"
echo "future3d_num_query_tokens=${future3d_num_query_tokens}"
echo "accelerate_config=${accelerate_config}"
echo "per_device_batch_size=${per_device_batch_size}"
echo "freeze_module_list=${freeze_module_list}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
set -euo pipefail
echo "Host=$(hostname)  SLURM_PROCID=$SLURM_PROCID"

accelerate launch \
  --config_file "'"${accelerate_config}"'" \
  --main_process_ip "'"${MASTER_ADDR}"'" \
  --main_process_port "'"${MASTER_PORT}"'" \
  --machine_rank "$SLURM_PROCID" \
  --num_machines "'"${SLURM_NNODES}"'" \
  --num_processes "'"${TOTAL_GPUS}"'" \
  starVLA/training/train_starvla.py \
  --config_yaml "'"${config_yaml}"'" \
  --framework.name "'"${Framework_name}"'" \
  --framework.qwenvl.base_vlm "'"${base_vlm}"'" \
  --framework.qwenvl.attn_implementation "'"${attn_implementation}"'" \
  --framework.future3d.da3_model_path_or_name "'"${da3_model_path}"'" \
  --framework.future3d.num_query_tokens "'"${future3d_num_query_tokens}"'" \
  --datasets.vla_data.per_device_batch_size "'"${per_device_batch_size}"'" \
  --datasets.vla_data.data_root_dir "'"${data_root}"'" \
  --datasets.vla_data.data_mix "'"${data_mix}"'" \
  --trainer.freeze_modules "'"${freeze_module_list}"'" \
  --trainer.max_train_steps 60000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000000 \
  --run_root_dir "'"${run_root_dir}"'" \
  --run_id "'"${run_id}"'" \
  --wandb_project MagicBot-2B-OFT3D \
  --wandb_entity zaleni
'
