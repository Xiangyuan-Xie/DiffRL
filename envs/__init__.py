# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .dummy_differentiable import DummyDifferentiableVecEnv

try:
    from .dflex_env import DFlexEnv
    from .ant import AntEnv
    from .cartpole_swing_up import CartPoleSwingUpEnv
    from .cheetah import CheetahEnv
    from .hopper import HopperEnv
    from .humanoid import HumanoidEnv
    from .snu_humanoid import SNUHumanoidEnv
except ImportError:
    DFlexEnv = None
    AntEnv = None
    CartPoleSwingUpEnv = None
    CheetahEnv = None
    HopperEnv = None
    HumanoidEnv = None
    SNUHumanoidEnv = None

__all__ = [
    "DummyDifferentiableVecEnv",
    "DFlexEnv",
    "AntEnv",
    "CartPoleSwingUpEnv",
    "CheetahEnv",
    "HopperEnv",
    "HumanoidEnv",
    "SNUHumanoidEnv",
]
