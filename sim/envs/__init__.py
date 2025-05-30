"""Registers the tasks in the task registry.

For other people who might be looking at this in the future - my preferred way
of doing config management is to use dataclasses (see the `mlfab` or `xax`
packages for examples of what I mean). This plays a lot better with type
checkers and VSCode. I am just doing it this way to get something working
quickly.
"""

# mypy: ignore-errors
from sim.envs.humanoids.dora_config import DoraCfg, DoraCfgPPO
from sim.envs.humanoids.dora_env import DoraFreeEnv
from sim.envs.humanoids.g1_config import G1Cfg, G1CfgPPO
from sim.envs.humanoids.g1_env import G1FreeEnv
from sim.envs.humanoids.h1_config import H1Cfg, H1CfgPPO
from sim.envs.humanoids.h1_env import H1FreeEnv
from sim.envs.humanoids.stompymicro_config import (
    StompyMicroCfg,
    StompyMicroCfgPPO,
    StompyMicroStandingCfg,
)
from sim.envs.humanoids.stompymicro_env import StompyMicroEnv
from sim.envs.humanoids.stompymini_config import MiniCfg, MiniCfgPPO
from sim.envs.humanoids.stompymini_env import MiniFreeEnv
from sim.envs.humanoids.stompypro_config import (
    StompyProCfg,
    StompyProCfgPPO,
    StompyProStandingCfg,
)
from sim.envs.humanoids.stompypro_env import StompyProFreeEnv

#zeroth01
from sim.envs.humanoids.zeroth01_config import Zeroth01Cfg, Zeroth01CfgPPO
from sim.envs.humanoids.zeroth01_env import Zeroth01Env

from sim.envs.humanoids.xbot_config import XBotCfg, XBotCfgPPO
from sim.envs.humanoids.xbot_env import XBotLFreeEnv
from sim.utils.task_registry import TaskRegistry  # noqa: E402

task_registry = TaskRegistry()
task_registry.register("stompymini", MiniFreeEnv, MiniCfg(), MiniCfgPPO())
task_registry.register("stompypro", StompyProFreeEnv, StompyProCfg(), StompyProCfgPPO())
task_registry.register("stompypro_standing", StompyProFreeEnv, StompyProStandingCfg(), StompyProCfgPPO())
task_registry.register("dora", DoraFreeEnv, DoraCfg(), DoraCfgPPO())
task_registry.register("h1", H1FreeEnv, H1Cfg(), H1CfgPPO())
task_registry.register("g1", G1FreeEnv, G1Cfg(), G1CfgPPO())
task_registry.register("XBotL_free", XBotLFreeEnv, XBotCfg(), XBotCfgPPO())
task_registry.register("stompymicro", StompyMicroEnv, StompyMicroCfg(), StompyMicroCfgPPO())
task_registry.register("stompymicro_standing", StompyMicroEnv, StompyMicroStandingCfg(), StompyMicroCfgPPO())
task_registry.register("zeroth01", Zeroth01Env, Zeroth01Cfg(), Zeroth01CfgPPO())
