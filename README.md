# MagicBot-2B-OFT-Robotwin

Evaluation guide for the released RoboTwin checkpoint:

- Hugging Face repo: `zaleni/MagicBot-2B-OFT-Robotwin`
- Codebase: this repository
- Benchmark: RoboTwin 2.0
- Recommended eval entry: `examples/Robotwin/eval_files/eval_robotwin_oft3d_11tasks.sh`

This checkpoint is evaluated through MagicBot's policy server plus a third-party RoboTwin simulator checkout.

## What You Need

Prepare these pieces before running evaluation:

- A Linux machine with NVIDIA GPU(s)
- Python 3.10
- This MagicBot repository
- A working RoboTwin checkout
- The released checkpoint snapshot from `zaleni/MagicBot-2B-OFT-Robotwin`
- The base VLM and DA3 weights referenced by the checkpoint config

One Python environment is enough. If you prefer, you can also use two environments:

- `STARVLA_PYTHON`: Python used for MagicBot / policy server
- `ROBOTWIN_PYTHON`: Python used for RoboTwin evaluation

For a simple setup, point both variables to the same Python interpreter.

## 1. Install MagicBot

```bash
git clone <your-magicbot-repo-url>
cd MagicBot-2B-OFT

conda create -n magicbot python=3.10 -y
conda activate magicbot

# Install a CUDA-matching PyTorch first if your machine doesn't have one yet.
# Then install MagicBot dependencies.
pip install -r requirements.txt
pip install -e .

# RoboTwin eval helper dependencies used by the wrapper scripts.
pip install -r examples/Robotwin/eval_files/requirements.txt

# Optional but recommended. If unavailable, the model loader will fall back to sdpa.
pip install flash-attn --no-build-isolation
```

## 2. Install RoboTwin

Clone and install RoboTwin by following the RoboTwin upstream instructions for simulator dependencies.

After installation, you should have a local checkout like:

```bash
/path/to/RoboTwin
```

During evaluation, export:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
```

## 3. Download the Released Checkpoint

Download the full snapshot from Hugging Face. Do not copy out only `pytorch_model.pt`.

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download \
  zaleni/MagicBot-2B-OFT-Robotwin \
  --local-dir ./checkpoints/MagicBot-2B-OFT-Robotwin
```

Expected layout:

```text
checkpoints/MagicBot-2B-OFT-Robotwin/
  config.yaml
  dataset_statistics.json
  final_model/
    pytorch_model.pt
```

Important:

- MagicBot loads `config.yaml` and `dataset_statistics.json` relative to the checkpoint file.
- Keep the directory structure intact.
- Pass the checkpoint file path itself, for example `.../final_model/pytorch_model.pt`.

## 4. Fix Local Paths in `config.yaml`

The released `config.yaml` may still contain absolute paths from the training machine. Before evaluation, open the downloaded `config.yaml` and update these fields to your local paths:

```yaml
framework:
  qwenvl:
    base_vlm: /path/to/your/Qwen3.5-2B
  future3d:
    da3_model_path_or_name: /path/to/your/DA3-LARGE-1.1
```

If these paths are not updated, you may see errors such as:

- `HFValidationError`
- local file not found for `base_vlm`
- local file not found for `da3_model_path_or_name`

## 5. Optional RoboTwin Compatibility Patch

The current MagicBot eval wrapper already tries to pass `policy_ckpt_path` safely. Most users can skip this step.

If your RoboTwin checkout still errors on `policy_ckpt_path`, patch `RoboTwin/script/eval_policy.py` so that it accepts an optional `--policy_ckpt_path` argument and forwards it into the loaded config:

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

## 6. Set Python Paths

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

## 7. Run Evaluation

### 1-GPU smoke test

```bash
cd MagicBot-2B-OFT

export CUDA_VISIBLE_DEVICES=0
export ROBOTWIN_PATH=/path/to/RoboTwin
export STARVLA_PYTHON=$CONDA_PREFIX/bin/python
export ROBOTWIN_PYTHON=$CONDA_PREFIX/bin/python

bash examples/Robotwin/eval_files/eval_robotwin_oft3d_11tasks.sh \
  -c ./checkpoints/MagicBot-2B-OFT-Robotwin/final_model/pytorch_model.pt \
  -n MagicBot-2B-OFT-Robotwin \
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
  -c ./checkpoints/MagicBot-2B-OFT-Robotwin/final_model/pytorch_model.pt \
  -n MagicBot-2B-OFT-Robotwin \
  -m randomized \
  -j 1 \
  -p 5694
```

Notes:

- `-m randomized` runs the randomized RoboTwin setting.
- `-j 1` means one concurrent eval job per visible GPU.
- The 11-task list is defined in `examples/Robotwin/eval_files/robotwin_oft3d_11tasks.txt`.

## 8. Where Results Are Saved

Results are written next to the checkpoint directory:

```text
<checkpoint_parent>/robotwin_eval_runs/<run_name>_<timestamp>/
```

For the command above, look for:

- `.../robotwin_eval_runs/<name>_<timestamp>/summary.txt`
- `.../robotwin_eval_runs/<name>_<timestamp>/summary.json`
- `.../robotwin_eval_runs/<name>_<timestamp>/demo_randomized/`

The `demo_randomized` folder contains per-task `*_server.log` and `*_eval.log`.

## 9. Common Issues

### `HFValidationError` or bad local model path

Cause:

- `config.yaml` still points to training-machine absolute paths

Fix:

- update `framework.qwenvl.base_vlm`
- update `framework.future3d.da3_model_path_or_name`

### `missing the required policy_ckpt_path patch`

Cause:

- your RoboTwin checkout is using an older `script/eval_policy.py`

Fix:

- use the current MagicBot eval scripts from this repo
- or apply the optional compatibility patch above

### `opening handshake failed` in `server.log`

Cause:

- the eval launcher is probing whether the websocket server port is ready

If evaluation is progressing and `Success rate:` lines continue to appear in `*_eval.log`, this message can usually be ignored.

## 10. Reference Files

- `examples/Robotwin/eval_files/eval_robotwin_oft3d_11tasks.sh`
- `examples/Robotwin/eval_files/start_eval.sh`
- `examples/Robotwin/eval_files/eval.sh`
- `examples/Robotwin/eval_files/run_policy_server.sh`
- `examples/Robotwin/eval_files/deploy_policy.yml`

