"""
This file implements a KSP environment for the detumbling task, 
building on Piotr Kubica's work on using kRPC to control KSP using Python.
"""

import math
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def get_sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0


class GameEnv(gym.Env):
    """A KSP environment for the three-axis detumbling task."""
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


class SimplerGameEnv(gym.Env):
    """A simpler version of the GameEnv with only one action to detumble along the roll axis."""
    def __init__(self, conn):
        self.set_telemetry(conn)
        self.pre_launch_setup()

        self.action_space = spaces.Discrete(9)  # -1, 0, 1 for all axis
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        # TODO(adrian) dict of dicts so we can unroll choose_action automatically
        self.available_actions = {
            "roll": (self.vessel.control.roll, (-1, 0, 1)),
        }

    def set_telemetry(self, conn):
        self.conn = conn
        self.vessel = conn.space_center.active_vessel

        # Setting up streams for telemetry
        self.ut = conn.add_stream(getattr, conn.space_center, "ut")
        self.pitch = conn.add_stream(getattr, self.vessel.flight(), "pitch")
        self.heading = conn.add_stream(getattr, self.vessel.flight(), "heading")
        self.roll = conn.add_stream(getattr, self.vessel.flight(), "roll")


    def pre_launch_setup(self):
        self.vessel.control.sas = True
        self.vessel.control.rcs = False

    def step(self, action):
        """
        possible continuous actions: yaw[-1:1], pitch[-1:1], roll[-1:1], throttle[0:1],
        other: forward[-1:1], up[-1:1], right[-1:1], wheel_throttle[-1:1], wheel_steering[-1:1],
        available observation
        https://krpc.github.io/krpc/python/api/space-center/control.html
        available states:
        https://krpc.github.io/krpc/python/api/space-center/flight.html
        https://krpc.github.io/krpc/python/api/space-center/orbit.html
        https://krpc.github.io/krpc/python/api/space-center/reference-frame.html
        :param action:
        :return state, reward, done, {}:
        """
        done = False

        start_act = self.ut()

        # n sticky actions in one second
        n = 3
        while self.ut() - start_act <= 1/n:
            self.choose_action(action)
        # for i in range(times_action_repeat):      # could warp time like this
            # self.conn.space_center.warp_to(start_act + (i + 1) * 1 / n)

        state = self.get_state()
        reward = self.compute_reward()
        self.conn.ui.message("Reward: " + str(round(reward, 2)), duration=0.5)

        return state, reward, False, {}

    def choose_action(self, action):

        self.vessel.control.roll = action - 1.0

        # self.conn.ui.message(
        #     f"Roll = {action - 1}", duration=0.5,
        # )

    def reset(self, seed):
        """
        :return: state
        """
        quick_save = "cubesat_undeployed_tumbling"

        try:
            self.conn.space_center.load(quick_save)
        except Exception as ex:
            print("Error:", ex)
            exit(f"You have no quick save named '{quick_save}'. Terminating.")

        time.sleep(3)

        # game is reloaded and we need to reset the telemetry
        self.set_telemetry(self.conn)
        self.pre_launch_setup()

        self.conn.space_center.physics_warp_factor = 0

        state = self.get_state()

        return state, {}

    def get_state(self):
        state = [
            math.sin(math.radians(self.roll())),
            math.cos(math.radians(self.roll())),
        ]
        return state

    def compute_reward(self):           # maximum reward at zero roll
        return -abs(self.roll()) / 180


class DummyEnv:
    """A dummy non-KSP environment for simply testing the maintenance of an angle."""
    def __init__(self, **config) -> None:
        self.angle = np.pi

    def get_state(self):
        return [math.sin(self.angle), math.cos(self.angle)]

    def reset(self, seed=None):
        return self.get_state()

    def step(self, action):
        torque = (action - 1.0) * 0.5

        self.angle += torque
        if self.angle < 0:
            self.angle += (2 * np.pi)
        self.angle %= (2 * np.pi)

        reward = -abs(np.pi - self.angle) / np.pi       # maximum reward along roll=pi
        return self.get_state(), reward, False, {}


if __name__ == "__main__":
    env = DummyEnv()
    for i in range(100):
        state, reward, _, _ = env.step(0)
        print(env.angle, reward)
