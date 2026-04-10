from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deployment.model_server.tools.image_tools import to_pil_preserve

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.future3d import (
    DA3BackboneTeacher,
    Future3DPerceiverResampler,
    resolve_da3_backbone_defaults,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenOFT3DDefaultConfig:
    name: str = "QwenOFT3D"

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3.5-2B",
        "attn_implementation": "flash_attention_2",
    })

    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MLP",
        "action_dim": 14,
        "action_hidden_dim": 2560,
        "future_action_window_size": 49,
        "past_action_window_size": 0,
        "placeholder_token": "\U0001F50D",
    })

    future3d: dict = field(default_factory=lambda: {
        "enable": True,
        "future_delta": 15,
        "num_query_tokens": 432,
        "placeholder_token": "\u25C6",
        "query_layer_indices": [11, 15, 19, 23],
        "lambda_3d": 0.01,
        "da3_model_path_or_name": "/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/DA3-LARGE-1-1",
        "da3_variant": "large",
        "da3_code_root": None,
        "da3_teacher_process_res": 504,
        "da3_teacher_layers": None,
        "da3_query_dim": None,
        "da3_tokens_per_view": 1296,
        "da3_num_views": 3,
        "da3_layer_weights": [1.0, 1.2, 1.4, 1.6],
        "future_query_init_std": 0.02,
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("QwenOFT3D")
class QwenOFT3D(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenOFT3DDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.action_token = self.config.framework.action_model.get("placeholder_token", "\U0001F50D")
        self.action_token_id = self._resolve_single_token_id(self.action_token, "action")
        self.l1_loss = nn.L1Loss()

        self.future3d_cfg = self.config.framework.future3d
        self.future3d_enabled = self.future3d_cfg.get("enable", True) not in ["False", False]
        self.lambda_3d = float(self.future3d_cfg.get("lambda_3d", 0.01))

        if self.future3d_enabled:
            da3_defaults = resolve_da3_backbone_defaults(
                self.future3d_cfg.da3_model_path_or_name,
                self.future3d_cfg.get("da3_variant", "auto"),
            )
            if self.future3d_cfg.get("da3_teacher_layers", None) is None:
                self.future3d_cfg.da3_teacher_layers = list(da3_defaults["teacher_layers"])
            if self.future3d_cfg.get("da3_query_dim", None) is None:
                self.future3d_cfg.da3_query_dim = int(da3_defaults["query_dim"])

            self.query_layer_indices = tuple(int(layer_idx) for layer_idx in self.future3d_cfg.query_layer_indices)
            self.da3_teacher_layers = tuple(int(layer_idx) for layer_idx in self.future3d_cfg.da3_teacher_layers)
            self.da3_layer_weights = tuple(float(weight) for weight in self.future3d_cfg.da3_layer_weights)
            if len(self.query_layer_indices) != len(self.da3_teacher_layers):
                raise ValueError("framework.future3d.query_layer_indices and da3_teacher_layers must have the same length")
            if len(self.query_layer_indices) != len(self.da3_layer_weights):
                raise ValueError("framework.future3d.da3_layer_weights must align with query_layer_indices")

            self.num_future_query_tokens = int(self.future3d_cfg.num_query_tokens)
            self.da3_num_views = int(self.future3d_cfg.da3_num_views)
            if self.num_future_query_tokens % self.da3_num_views != 0:
                raise ValueError("framework.future3d.num_query_tokens must be divisible by da3_num_views")
            self.future_tokens_per_view = self.num_future_query_tokens // self.da3_num_views
            self.da3_tokens_per_view = int(self.future3d_cfg.da3_tokens_per_view)
            self.da3_query_dim = int(self.future3d_cfg.da3_query_dim)
            self.future_3d_token = self.future3d_cfg.get("placeholder_token", "\u25C6")
            self.future_3d_token_id = self._resolve_single_token_id(self.future_3d_token, "future3d")
            if self.future_3d_token_id == self.action_token_id:
                raise ValueError("future3d placeholder token must be different from action placeholder token")

            hidden_size = self.qwen_vl_interface.model.config.hidden_size
            num_attention_heads = getattr(self.qwen_vl_interface.model.config, "num_attention_heads", None)
            if num_attention_heads is None:
                num_attention_heads = getattr(self.qwen_vl_interface.model.config.text_config, "num_attention_heads", 16)
            init_std = float(self.future3d_cfg.get("future_query_init_std", 0.02))

            self.future_3d_queries = nn.Parameter(torch.randn(1, self.num_future_query_tokens, hidden_size) * init_std)
            self.future_3d_output_queries = nn.Parameter(
                torch.randn(1, self.da3_tokens_per_view, hidden_size) * init_std
            )
            self.future_3d_messenger_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_size) for _ in self.query_layer_indices]
            )
            self.future_3d_output_decoder = Future3DPerceiverResampler(
                dim=hidden_size,
                num_heads=int(num_attention_heads),
                output_dim=self.da3_query_dim,
            )
            self.da3_teacher = DA3BackboneTeacher(
                model_path_or_name=self.future3d_cfg.da3_model_path_or_name,
                process_res=int(self.future3d_cfg.get("da3_teacher_process_res", 504)),
                dtype=torch.bfloat16,
                teacher_layers=self.da3_teacher_layers,
                code_root=self.future3d_cfg.get("da3_code_root", None),
            )
            if self.da3_teacher.feature_dim != self.da3_query_dim:
                raise ValueError(
                    f"DA3 teacher dim ({self.da3_teacher.feature_dim}) does not match da3_query_dim ({self.da3_query_dim})"
                )
        else:
            self.query_layer_indices = tuple()
            self.da3_teacher_layers = tuple()
            self.da3_layer_weights = tuple()
            self.num_future_query_tokens = 0
            self.da3_num_views = 0
            self.future_tokens_per_view = 0
            self.da3_tokens_per_view = 0
            self.da3_query_dim = 0
            self.future_3d_token = None
            self.future_3d_token_id = None
            self.register_parameter("future_3d_queries", None)
            self.register_parameter("future_3d_output_queries", None)
            self.future_3d_messenger_norms = nn.ModuleList()
            self.future_3d_output_decoder = None
            self.da3_teacher = None

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        qwenvl_outputs, future_positions, action_positions = self._run_qwen(batch_images, instructions)
        last_hidden = qwenvl_outputs.hidden_states[-1]

        action_queries = self._gather_positions(last_hidden, action_positions).float()
        pred_actions = self.action_model.predict_action(action_queries)
        actions = torch.as_tensor(np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype)
        actions_target = actions[:, -self.chunk_len :, :]
        action_loss = self.l1_loss(pred_actions, actions_target)

        output_dict = {"action_loss": action_loss}
        total_loss = action_loss

        if self.future3d_enabled and self.lambda_3d > 0:
            future_images, future_image_mask = self._build_future_image_batch(examples, pred_actions.device)
            loss_3d, loss_logs = self.compute_3d_query_loss(
                hidden_states=qwenvl_outputs.hidden_states,
                future_positions=future_positions,
                future_images=future_images,
                img_masks=future_image_mask,
            )
            output_dict["loss_3d"] = loss_3d
            output_dict.update(loss_logs)
            total_loss = action_loss + self.lambda_3d * loss_3d

        output_dict["total_loss"] = total_loss
        return output_dict

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwenvl_outputs, _, action_positions = self._run_qwen(batch_images, instructions)
        last_hidden = qwenvl_outputs.hidden_states[-1]
        action_queries = self._gather_positions(last_hidden, action_positions).float()
        pred_actions = self.action_model.predict_action(action_queries)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _run_qwen(self, batch_images, instructions):
        instructions = [instruction + self._build_prompt_suffix() for instruction in instructions]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = qwen_inputs["input_ids"]

        future_positions = None
        action_positions = self._select_token_positions(input_ids, self.action_token_id, self.chunk_len, "action")
        hook_handle = None

        if self.future3d_enabled:
            future_positions = self._select_token_positions(
                input_ids,
                self.future_3d_token_id,
                self.num_future_query_tokens,
                "future3d",
            )
            embedding_layer = self._get_embedding_layer()
            batch_size = input_ids.shape[0]

            def inject_future_query_hook(module, inputs, output):
                query_embed = self.future_3d_queries.expand(batch_size, -1, -1).to(dtype=output.dtype, device=output.device)
                batch_indices = torch.arange(batch_size, device=output.device).unsqueeze(1).expand_as(future_positions)
                output[batch_indices, future_positions.to(output.device), :] = query_embed
                return output

            hook_handle = embedding_layer.register_forward_hook(inject_future_query_hook)

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        return qwenvl_outputs, future_positions, action_positions

    def _build_prompt_suffix(self) -> str:
        action_tokens = self.action_token * self.chunk_len
        if not self.future3d_enabled:
            return f" Predict the next {self.chunk_len} robot actions with <action>{action_tokens}</action>."

        future_tokens = self.future_3d_token * self.num_future_query_tokens
        return (
            f" First encode the future 3D scene with <future3d>{future_tokens}</future3d>. "
            f"Then predict the next {self.chunk_len} robot actions with <action>{action_tokens}</action>."
        )

    def _get_embedding_layer(self):
        embedding_layer = None
        if hasattr(self.qwen_vl_interface.model, "get_input_embeddings"):
            embedding_layer = self.qwen_vl_interface.model.get_input_embeddings()
        if embedding_layer is None and hasattr(self.qwen_vl_interface.model, "model"):
            embedding_layer = self.qwen_vl_interface.model.model.get_input_embeddings()
        if embedding_layer is None:
            raise RuntimeError("Failed to locate the Qwen input embedding layer for future-3D query injection.")
        return embedding_layer

    def _resolve_single_token_id(self, token: str, token_name: str) -> int:
        token_ids = self.qwen_vl_interface.processor.tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"{token_name} placeholder token {token!r} is tokenized into {len(token_ids)} pieces: {token_ids}. "
                "Please choose a placeholder that maps to a single tokenizer id."
            )
        return token_ids[0]

    def _select_token_positions(
        self,
        input_ids: torch.Tensor,
        token_id: int,
        expected_count: int,
        token_name: str,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        mask = input_ids == token_id
        counts = mask.sum(dim=1)
        if (counts < expected_count).any():
            insufficient = (counts < expected_count).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient {token_name} tokens (< {expected_count}): "
                f"{insufficient} | counts={counts.tolist()}"
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        masked_positions = torch.where(mask, positions, torch.full_like(positions, -1))
        topk_positions = masked_positions.topk(k=expected_count, dim=-1).values
        return topk_positions.sort(dim=-1).values

    def _gather_positions(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device).unsqueeze(1).expand_as(positions)
        return hidden_states[batch_indices, positions.to(hidden_states.device)]

    def _build_future_image_batch(self, examples: List[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        future_images = []
        future_image_masks = []
        for example in examples:
            if "future_image" not in example:
                raise KeyError("QwenOFT3D requires `future_image` in each training sample.")

            view_tensors = []
            for image in example["future_image"]:
                image_array = np.asarray(image)
                if image_array.ndim != 3:
                    raise ValueError(f"Expected future image with shape [H, W, C], got {image_array.shape}")
                view_tensors.append(torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1))
            future_images.append(torch.stack(view_tensors, dim=0))
            future_image_masks.append(torch.as_tensor(example.get("future_image_mask", [True] * len(view_tensors)), dtype=torch.bool))

        future_images = torch.stack(future_images, dim=0).to(device=device, dtype=torch.float32)
        future_image_masks = torch.stack(future_image_masks, dim=0).to(device=device, dtype=torch.bool)
        return future_images, future_image_masks

    def _project_query_layers(self, hidden_states: Tuple[torch.Tensor, ...], future_positions: torch.Tensor) -> list[torch.Tensor]:
        if self.future_3d_output_decoder is None:
            return []

        decoder_param = next(self.future_3d_output_decoder.parameters())
        decoder_device = decoder_param.device
        decoder_dtype = decoder_param.dtype
        batch_size = future_positions.shape[0]
        output_queries = self.future_3d_output_queries.expand(batch_size, -1, -1).to(
            device=decoder_device,
            dtype=decoder_dtype,
        )
        output_queries = output_queries[:, None, :, :].expand(batch_size, self.da3_num_views, -1, -1)
        output_queries = output_queries.reshape(batch_size * self.da3_num_views, self.da3_tokens_per_view, -1)

        projected_queries = []
        for layer_idx, layer_hidden in enumerate(hidden_states):
            query_tokens = self._gather_positions(layer_hidden, future_positions).to(
                device=decoder_device,
                dtype=decoder_dtype,
            )
            query_tokens = self.future_3d_messenger_norms[layer_idx](query_tokens)
            expected_query_tokens = self.da3_num_views * self.future_tokens_per_view
            if query_tokens.shape[1] != expected_query_tokens:
                raise ValueError(f"Expected {expected_query_tokens} future query tokens, got {query_tokens.shape[1]}")

            query_tokens = query_tokens.reshape(batch_size, self.da3_num_views, self.future_tokens_per_view, -1)
            messenger_tokens = query_tokens.reshape(batch_size * self.da3_num_views, self.future_tokens_per_view, -1)
            decoded_queries = self.future_3d_output_decoder(output_queries, messenger_tokens)
            decoded_queries = decoded_queries.reshape(batch_size, self.da3_num_views * self.da3_tokens_per_view, -1)
            projected_queries.append(decoded_queries)
        return projected_queries

    def get_3d_token_mask(self, img_masks: torch.Tensor, target_len: int) -> torch.Tensor:
        token_mask = img_masks.unsqueeze(-1).expand(-1, -1, self.da3_tokens_per_view).reshape(img_masks.shape[0], -1)
        if token_mask.shape[1] == target_len:
            return token_mask
        token_mask = token_mask[:, None, :].to(dtype=torch.float32)
        token_mask = F.interpolate(token_mask, size=target_len, mode="nearest")
        return token_mask[:, 0, :].to(dtype=torch.bool)

    def prepare_da3_teacher_inputs(self, future_images: torch.Tensor, img_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid_view_masks = img_masks.to(dtype=torch.bool)
        valid_view_counts = valid_view_masks.sum(dim=1)
        incomplete_samples = (valid_view_counts > 0) & (valid_view_counts < future_images.shape[1])
        if not incomplete_samples.any():
            return future_images, valid_view_masks

        num_views = future_images.shape[1]
        batch_indices = torch.arange(future_images.shape[0], device=future_images.device)
        primary_view_indices = valid_view_masks.to(dtype=torch.int64).argmax(dim=1)
        primary_images = future_images[batch_indices, primary_view_indices]

        teacher_images = future_images.clone()
        invalid_view_masks = ~valid_view_masks
        if invalid_view_masks.any():
            teacher_images[invalid_view_masks] = primary_images.unsqueeze(1).expand(-1, num_views, -1, -1, -1)[
                invalid_view_masks
            ]
        return teacher_images, valid_view_masks.clone()

    def compute_3d_query_loss(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        future_positions: torch.Tensor,
        future_images: torch.Tensor,
        img_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.da3_teacher is None or future_positions is None:
            zero = future_images.new_zeros((), dtype=torch.float32)
            return zero, {}

        if len(hidden_states) <= max(self.query_layer_indices) + 1:
            raise ValueError(
                f"Qwen hidden state count ({len(hidden_states) - 1} layers) is insufficient for query layers "
                f"{self.query_layer_indices}"
            )

        selected_hidden_states = tuple(hidden_states[layer_idx + 1] for layer_idx in self.query_layer_indices)
        projected_queries = self._project_query_layers(selected_hidden_states, future_positions)
        teacher_images, teacher_img_masks = self.prepare_da3_teacher_inputs(future_images, img_masks)
        with torch.no_grad():
            teacher_layers = self.da3_teacher(teacher_images)

        token_mask = self.get_3d_token_mask(teacher_img_masks, teacher_layers[0].shape[1])
        total_loss = future_images.new_zeros((), dtype=torch.float32)
        loss_logs = {}
        valid_layer_count = 0

        for pred, target, weight, teacher_layer_idx, query_layer_idx in zip(
            projected_queries,
            teacher_layers,
            self.da3_layer_weights,
            self.da3_teacher_layers,
            self.query_layer_indices,
            strict=False,
        ):
            target = target.to(device=pred.device, dtype=pred.dtype)
            pred_valid = pred[token_mask]
            target_valid = target[token_mask]
            if pred_valid.numel() == 0:
                continue

            pred_norm = F.normalize(pred_valid, p=2, dim=-1)
            target_norm = F.normalize(target_valid.detach(), p=2, dim=-1)
            cos_loss = (1.0 - (pred_norm * target_norm).sum(dim=-1)).mean()

            pred_ln = F.layer_norm(pred_valid, normalized_shape=(pred_valid.shape[-1],))
            target_ln = F.layer_norm(target_valid.detach(), normalized_shape=(target_valid.shape[-1],))
            mse_loss = F.mse_loss(pred_ln, target_ln)

            layer_loss = (cos_loss + mse_loss) * weight
            total_loss = total_loss + layer_loss
            loss_logs[f"loss_3d_q{query_layer_idx}_t{teacher_layer_idx}"] = layer_loss.detach()
            valid_layer_count += 1

        if valid_layer_count > 0:
            total_loss = total_loss / valid_layer_count
        else:
            total_loss = total_loss * 0.0
        return total_loss, loss_logs
