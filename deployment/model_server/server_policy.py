# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    from_pretrained_kwargs = {}
    if args.disable_3d_teacher_for_eval:
        from_pretrained_kwargs = {
            "config_overrides": {
                "framework": {
                    "future3d": {
                        "load_training_only_modules": False,
                        "lambda_3d": 0.0,
                    }
                }
            },
            "load_state_strict": False,
            "state_dict_skip_prefixes": (
                "da3_teacher.",
                "future_3d_output_queries",
                "future_3d_messenger_norms.",
                "future_3d_output_decoder.",
            ),
        }
        logging.info("Loading policy in eval-optimized OFT3D mode: skipping 3D loss-only modules.")

    vla = baseframework.from_pretrained(  # TODO should auto detect framework from model path
        args.ckpt_path,
        **from_pretrained_kwargs,
    )

    if args.use_bf16:  # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--disable_3d_teacher_for_eval",
        action="store_true",
        help="Skip loading OFT3D training-only 3D loss modules during inference/eval.",
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
