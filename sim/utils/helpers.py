# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# mypy: disable-error-code="no-untyped-def"
# type: ignore
import argparse
import copy
import os
import random
from typing import Any, Tuple, Union

from datetime import datetime

import numpy as np

from isaacgym import gymapi, gymutil  # isort: skip

import torch  # isort: skip


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args() -> argparse.Namespace:
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "stompymicro",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
        {
            "name": "--log_h5",
            "action": "store_true",
            "default": False,
            "help": "Enable HDF5 logging",
        },
        {
            "name": "--command-arrow",
            "action": "store_true",
            "default": False,
            "help": "Draw command and velocity arrows during visualization",
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic: Any, path: Union[str, os.PathLike]) -> None:
    os.makedirs(path, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    print("当前时间:", current_time)
    path = os.path.join(path, current_time + "_policy.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def export_policy_as_onnx(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    print("当前时间:", current_time)
    path = os.path.join(path, current_time + "_policy_1.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")

    # Get the input dimension from the first layer of the model
    first_layer = next(model.parameters())
    input_dim = first_layer.shape[1]

    # Create a dummy input tensor with the correct dimensions
    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(model, dummy_input, path)


def draw_vector(
    gym: gymapi.Gym,
    viewer: gymapi.Viewer,
    env_handle: gymapi.Env,
    start_pos: np.ndarray,
    direction: Tuple[float, float],
    color: Tuple[float, float, float],
    clear_lines: bool = False,
    head_scale: float = 0.1,
) -> None:
    """Draws a single vector with an arrowhead."""
    if viewer is None:
        return

    # Unpack direction and create start position
    vel_x, vel_y = direction
    start = gymapi.Vec3(start_pos[0], start_pos[1], start_pos[2])

    # Scale arrow length by magnitude
    vel_magnitude = np.sqrt(vel_x**2 + vel_y**2)
    if vel_magnitude > 0:
        arrow_scale = np.clip(vel_magnitude, 0.1, 1.0)
        normalized_x = vel_x / vel_magnitude
        normalized_y = vel_y / vel_magnitude
    else:
        arrow_scale = 0.1
        normalized_x = 0
        normalized_y = 0

    # Calculate end position and arrowhead
    end = gymapi.Vec3(start.x + normalized_x * arrow_scale, start.y + normalized_y * arrow_scale, start.z)

    # Calculate perpendicular vector for arrowhead
    perp_x = -normalized_y
    perp_y = normalized_x

    head_left = gymapi.Vec3(
        end.x - head_scale * (normalized_x * 0.7 + perp_x * 0.7),
        end.y - head_scale * (normalized_y * 0.7 + perp_y * 0.7),
        end.z,
    )

    head_right = gymapi.Vec3(
        end.x - head_scale * (normalized_x * 0.7 - perp_x * 0.7),
        end.y - head_scale * (normalized_y * 0.7 - perp_y * 0.7),
        end.z,
    )

    # Create vertices and colors
    verts = [
        start.x,
        start.y,
        start.z,
        end.x,
        end.y,
        end.z,
        end.x,
        end.y,
        end.z,
        head_left.x,
        head_left.y,
        head_left.z,
        end.x,
        end.y,
        end.z,
        head_right.x,
        head_right.y,
        head_right.z,
    ]
    colors = [color[0], color[1], color[2]] * 6

    if clear_lines:
        gym.clear_lines(viewer)
    gym.add_lines(viewer, env_handle, 3, verts, colors)
