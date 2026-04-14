# MagicBot-2B-OFT-Robotwin

This README focuses on the stable workflow for RoboTwin:

1. prepare the environment
2. train your own MagicBot RoboTwin model
3. run RoboTwin inference / evaluation on the produced checkpoint

The public checkpoint may require local path fixes before it can be used directly, so the instructions below are written around locally produced checkpoints.

## What You Need

Prepare these items first:

- a Linux machine with NVIDIA GPU(s)
- Python 3.10
- this MagicBot repository
- a working RoboTwin checkout
- RoboTwin training data
- the base VLM weights used by the training script
- the DA3 weights used by OFT3D

Recommended base assets:

- `Qwen3.5-2B`
- `DA3-LARGE-1.1`

One Python environment is enough. If you want, you can also use two:

- `STARVLA_PYTHON` for the MagicBot policy server
- `ROBOTWIN_PYTHON` for RoboTwin evaluation

For the simplest setup, point both variables to the same Python interpreter.

## 1. Install MagicBot

```bash
git clone <your-magicbot-repo-url>
cd MagicBot-2B-OFT

conda create -n magicbot python=3.10 -y
conda activate magicbot

# Install a CUDA-matching PyTorch first if needed.
pip install -r requirements.txt
pip install -e .

# RoboTwin eval helper dependencies used by the wrapper scripts.
pip install -r examples/Robotwin/eval_files/requirements.txt

# Optional but recommended. If flash-attn is unavailable, the loader will fall back to sdpa.
pip install flash-attn --no-build-isolation
```

## 2. Install RoboTwin

Clone RoboTwin and finish its simulator installation by following the RoboTwin upstream instructions.

Example:

```bash
/path/to/RoboTwin
```

During evaluation, export:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
```

## 3. Optional RoboTwin Compatibility Patch

If your RoboTwin checkout is older, evaluation may fail with a `policy_ckpt_path` error. In that case, patch `RoboTwin/script/eval_policy.py` so it accepts an optional `--policy_ckpt_path` argument and forwards it into the loaded config:

```diff
@@
-    policy_ckpt_path = usr_args["policy_ckpt_path"]
+    policy_ckpt_path = usr_args.get("policy_ckpt_path")
@@
-    args["policy_ckpt_path"] = policy_ckpt_path
+    if policy_ckpt_path is not None:
+        args["policy_ckpt_path"] = policy_ckpt_path
@@
-    parser.add_argument("--policy_ckpt_path", type=str, required=True)
+    parser.add_argument("--policy_ckpt_path", type=str, default=None)
@@
-    config["policy_ckpt_path"] = args.policy_ckpt_path
+    if args.policy_ckpt_path is not None:
+        config["policy_ckpt_path"] = args.policy_ckpt_path
```

## 4. Prepare Training Paths

For training, edit one of these scripts:

- `examples/Robotwin/train_files/run_robotwin_train.sh`
- `examples/Robotwin/train_files/run_robotwin_train_slurm.sh`

At minimum, update these fields to your local paths:

- `base_vlm`
- `da3_model_path`
- `data_root`
- `run_root_dir`
- `run_id`

If you use the SLURM warm-start example, also check:

- `pretrained_checkpoint`

The default Robotwin config used by these scripts is:

- `examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml`

The key dataset settings in that config are:

- `data_mix: robotwin_selected_50_future3d`
- `action_mode: abs`
- `save_inference_only_weights: true`

## 5. Start Training

### Single-node 8-GPU training

After editing `run_robotwin_train.sh`, run:

```bash
conda activate magicbot
cd MagicBot-2B-OFT

bash examples/Robotwin/train_files/run_robotwin_train.sh
```

### SLURM 8-GPU training

After editing `run_robotwin_train_slurm.sh`, run:

```bash
conda activate magicbot
cd MagicBot-2B-OFT

sbatch examples/Robotwin/train_files/run_robotwin_train_slurm.sh
```

## 6. Training Outputs

Training outputs are written under:

```text
<run_root_dir>/<run_id>/
```

Expected layout:

```text
<run_root_dir>/<run_id>/
  config.yaml
  dataset_statistics.json
  checkpoints/
    steps_5000_pytorch_model.pt
    steps_10000_pytorch_model.pt
    ...
  final_model/
    pytorch_model.pt
```

Recommended evaluation targets:

- `.../checkpoints/steps_<N>_pytorch_model.pt`
- `.../final_model/pytorch_model.pt`

Important:

- Do not move only `pytorch_model.pt` somewhere else by itself.
- MagicBot also reads `config.yaml` and `dataset_statistics.json` relative to the checkpoint file.

## 7. Set Python Paths for Inference / Evaluation

Single-environment setup:

```bash
conda activate magicbot

export STARVLA_PYTHON=$CONDA_PREFIX/bin/python
export ROBOTWIN_PYTHON=$CONDA_PREFIX/bin/python
```

Two-environment setup:

```bash
export STARVLA_PYTHON=/path/to/starvla-env/bin/python
export ROBOTWIN_PYTHON=/path/to/robotwin-env/bin/python
```

## 8. If You Move the Checkpoint to Another Machine

Before evaluation, open the checkpoint's `config.yaml` and update these fields to local paths:

```yaml
framework:
  qwenvl:
    base_vlm: /path/to/your/Qwen3.5-2B
  future3d:
    da3_model_path_or_name: /path/to/your/DA3-LARGE-1.1
```

If these are not updated, you may see path-related errors such as:

- `HFValidationError`
- local file not found for `base_vlm`
- local file not found for `da3_model_path_or_name`

## 9. Run RoboTwin Inference / Evaluation

### 1-GPU smoke test

```bash
cd MagicBot-2B-OFT

export CUDA_VISIBLE_DEVICES=0
export ROBOTWIN_PATH=/path/to/RoboTwin
export STARVLA_PYTHON=$CONDA_PREFIX/bin/python
export ROBOTWIN_PYTHON=$CONDA_PREFIX/bin/python

bash examples/Robotwin/eval_files/eval_robotwin_oft3d_11tasks.sh \
  -c /path/to/<run_root_dir>/<run_id>/final_model/pytorch_model.pt \
  -n <run_id> \
  -m randomized \
  -j 1 \
  -p 5694
```

### 8-GPU randomized evaluation

```bash
cd MagicBot-2B-OFT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROBOTWIN_PATH=/path/to/RoboTwin
export STARVLA_PYTHON=$CONDA_PREFIX/bin/python
export ROBOTWIN_PYTHON=$CONDA_PREFIX/bin/python

bash examples/Robotwin/eval_files/eval_robotwin_oft3d_11tasks.sh \
  -c /path/to/<run_root_dir>/<run_id>/final_model/pytorch_model.pt \
  -n <run_id> \
  -m randomized \
  -j 1 \
  -p 5694
```

Notes:

- `-m randomized` runs the randomized RoboTwin setting
- `-j 1` means one concurrent eval job per visible GPU
- the 11-task list is defined in `examples/Robotwin/eval_files/robotwin_oft3d_11tasks.txt`

## 10. Where Evaluation Results Are Saved

Evaluation summaries are written next to the checkpoint directory:

```text
<checkpoint_parent>/robotwin_eval_runs/<run_name>_<timestamp>/
```

Look for:

- `.../summary.txt`
- `.../summary.json`
- `.../demo_randomized/`

The `demo_randomized` folder contains per-task:

- `*_server.log`
- `*_eval.log`

## 11. Where Videos Are Saved

RoboTwin videos are not written into the MagicBot log directory.

They are written under the RoboTwin checkout because the evaluation script changes into `ROBOTWIN_PATH` before launching RoboTwin.

The save pattern is:

```text
${ROBOTWIN_PATH}/eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/
```

Video filenames look like:

```text
episode0.mp4
episode1.mp4
...
```

If you cannot find videos, check whether `eval_video_log` is enabled in the RoboTwin task config. If it is off, evaluation will still run, but no mp4 files will be generated.

## 12. Common Issues

### `missing the required policy_ckpt_path patch`

Cause:

- your RoboTwin checkout is using an older `script/eval_policy.py`

Fix:

- use the current MagicBot eval scripts from this repo
- or apply the compatibility patch above

### `HFValidationError` or wrong local model path

Cause:

- the checkpoint `config.yaml` still points to paths from a different machine

Fix:

- update `framework.qwenvl.base_vlm`
- update `framework.future3d.da3_model_path_or_name`

### `opening handshake failed` in `server.log`

Cause:

- the eval launcher is probing whether the websocket server port is ready

If evaluation is progressing and `Success rate:` lines continue to appear in `*_eval.log`, this message can usually be ignored.

### No video files were generated

Cause:

- `eval_video_log` is disabled in RoboTwin

Fix:

- enable video logging in the RoboTwin task config
- then rerun evaluation

## 13. Reference Files

- `examples/Robotwin/train_files/run_robotwin_train.sh`
- `examples/Robotwin/train_files/run_robotwin_train_slurm.sh`
- `examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml`
- `examples/Robotwin/eval_files/eval_robotwin_oft3d_11tasks.sh`
- `examples/Robotwin/eval_files/start_eval.sh`
- `examples/Robotwin/eval_files/eval.sh`
- `examples/Robotwin/eval_files/run_policy_server.sh`
- `examples/Robotwin/eval_files/deploy_policy.yml`

