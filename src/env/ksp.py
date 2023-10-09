import datetime as dt
import math
import os
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def get_sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0


class GameEnv(gym.Env):
    def __init__(self, conn, wandb_logger):
        self.set_telemetry(conn)
        self.pre_launch_setup()
        self.wandb_logger = wandb_logger

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )

    def set_telemetry(self, conn):
        self.conn = conn
        self.vessel = conn.space_center.active_vessel

        self.ut = conn.add_stream(getattr, conn.space_center, "ut")
        self.pitch = conn.add_stream(getattr, self.vessel.flight(), "pitch")
        self.heading = conn.add_stream(getattr, self.vessel.flight(), "heading")
        self.roll = conn.add_stream(getattr, self.vessel.flight(), "roll")
        self.frame = self.vessel.orbit.body.reference_frame

        # TODO(adrian): extract rotational velocity or something to measure "tumbling"
        self.parts = conn.add_stream(getattr, self.vessel.parts, "all")

    def pre_launch_setup(self):
        self.vessel.control.sas = True
        self.vessel.control.rcs = False

    def step(self, action):
        done = False

        self.choose_action(action)
        # self.cheat_action()

        start_act = self.ut()

        while self.ut() - start_act <= 1 / 20:  # 20hz
            continue

        state = self.get_state()

        reward = self.compute_reward()

        self.conn.ui.message("Reward: " + str(round(reward, 2)), duration=0.5)
        self.wandb_logger.log({"reward": reward})

        return state, reward, done, done, {"mean_reward": reward}

    def cheat_action(self):
        self.vessel.control.roll = (
            get_sign(self.roll()) * math.log(abs(self.roll()) + 0.00001) * 0.1
        )
        self.vessel.control.pitch = 0
        self.vessel.control.yaw = 0

        self.conn.ui.message(
            f"{f'{self.roll():0.1f}'}",
            duration=1 / 20,
        )

    def choose_action(self, action):
        self.vessel.control.roll = action[0]
        self.vessel.control.pitch = action[1]
        self.vessel.control.yaw = action[2]

        self.conn.ui.message(
            f"{[f'{v:0.1f}' for v in action.tolist()]}",
            duration=1 / 20,
        )

    def reset(self, seed):
        quick_save = "sat1"

        self.conn.space_center.load(quick_save)

        time.sleep(3)

        # game is reloaded and we need to reset the telemetry
        self.set_telemetry(self.conn)
        self.pre_launch_setup()

        self.conn.space_center.physics_warp_factor = 0

        state = self.get_state()

        return state, {"conn": self.conn, "quicksave": "sat1"}

    def get_state(self):
        state = [
            math.sin(self.roll()),
            math.cos(self.roll()),
            math.sin(self.pitch()),
            math.cos(self.pitch()),
            math.sin(self.heading()),
            math.cos(self.heading()),
        ]

        return state

    def compute_reward(self):
        return 1 - abs(self.roll()) / 180
