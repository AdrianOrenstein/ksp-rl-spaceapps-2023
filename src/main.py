import glob
import math
from pathlib import Path
import time
from typing import Callable, Tuple
from gymnasium import spaces
from torch import nn
import torch

from env.ksp_env import GameEnv
import krpc
import numpy as np
from stable_baselines3 import SAC, PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy

run = wandb.init(
    project="detumbling",
    group="spaceapps-ksp-ai",
    monitor_gym=True,
    sync_tensorboard=True,
)

log_dir = list(glob.glob(f"./wandb/*-{run.id}"))
assert len(log_dir) == 1
log_dir = Path(log_dir[0])

env = GameEnv(krpc.connect(name="Tracker"), run)

model = SAC(
    "MlpPolicy",
    env=env,
    tensorboard_log=log_dir / "logs",
    # learning_rate=0.0003,
    # n_steps=128,
    # stats_window_size=128,
    device="cpu",
    verbose=1,
    buffer_size=1_000_000,
    learning_starts=128,
    batch_size=1024,
    train_freq=(32, "step"),
    stats_window_size=256,
)

model._last_obs = None

model.learn(
    total_timesteps=1_000_000,
    log_interval=1,
    progress_bar=True,
    reset_num_timesteps=False,
    callback=WandbCallback(),
)
