import collections
from datetime import datetime
import os
from typing import cast

import gymnasium as gym
from omegaconf import DictConfig

from pokemonred_puffer.environment import RedGymEnv


class CoordinatesWriter(gym.Wrapper):
    def __init__(self, env: RedGymEnv, config: DictConfig):
        super().__init__(env)
        self.coord_list = collections.deque()
        self.output_dir: str = config.output_dir
        self.step_counter = 0
        self.write_frequency: int = config.write_frequency
        self.write_path = os.path.join(
            self.output_dir,
            str(cast(RedGymEnv, self.env).env_id)
            + "-"
            + datetime.now().strftime("%Y%m%d%H%M%S")
            + "-coords.csv",
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.writer = open(self.write_path, "w")
        self.writer.write("")

    def step(self, action):
        # we run the step first so we can evaluate the result
        res = self.env.step(action)

        map_n = self.env.unwrapped.read_m("wCurMap")
        y_pos = self.env.unwrapped.read_m("wYCoord")
        x_pos = self.env.unwrapped.read_m("wXCoord")
        self.coord_list.append(
            [datetime.now().strftime("%Y%m%d%H%M%S"), str(map_n), str(y_pos), str(x_pos)]
        )
        if len(self.coord_list) >= self.write_frequency:
            self.write()
            self.step_counter = 0

        self.step_counter += 1

        return res

    def reset(self, *args, **kwargs):
        self.write()
        return self.env.reset(*args, **kwargs)

    def close(self):
        self.write()
        self.writer.close()
        return self.env.close()

    def write(self):
        self.writer.writelines(",".join(coord) + "\n" for coord in self.coord_list)
        self.coord_list.clear()


class ActionsWriter(gym.Wrapper):
    def __init__(self, env: RedGymEnv, config: DictConfig):
        super().__init__(env)
        self.action_list = collections.deque()
        self.output_dir: str = config.output_dir
        self.write_frequency: int = config.write_frequency
        self.write_path = os.path.join(
            self.output_dir, str(cast(RedGymEnv, self.env).env_id) + "-actions.csv"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.writer = open(self.write_path, "w")
        self.writer.write("")

    def step(self, action):
        self.action_list.append(action)
        if len(self.action_list) >= self.write_frequency:
            self.write()

        return self.env.step(action)

    def reset(self, *args, **kwargs):
        self.write()
        return self.env.reset(*args, **kwargs)

    def close(self):
        self.write()
        self.writer.close()
        return self.env.close()

    def write(self):
        self.writer.writelines(str(action) + "\n" for action in self.action_list)
        self.action_list.clear()
