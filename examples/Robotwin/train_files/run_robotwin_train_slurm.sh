# Optional NCCL NIC/HCA pinning. Set these in your shell if the cluster needs them.
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT3D
freeze_module_list=''
base_vlm=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Qwen3.5-2B
da3_model_path=${DA3_MODEL_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/DA3-LARGE-1-1}
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
run_root_dir=./results/Checkpoints
data_root=/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robotwin_all_50
data_mix=robotwin_selected_50_future3d
run_id=${data_mix}_qwen35_2b_oft3d_10task
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=offline

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.future3d.da3_model_path_or_name ${da3_model_path} \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 90000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 25 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project MagicBot-2B-OFT3D \
  --wandb_entity zaleni \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
