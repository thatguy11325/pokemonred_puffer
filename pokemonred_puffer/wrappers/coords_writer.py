import collections
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
        self.record_interval = 500
        self.step_counter = 0
        self.write_frequency: int = config.write_frequency
        self.write_path = os.path.join(
            self.output_dir, str(cast(RedGymEnv, self.env).env_id) + "-coords.csv"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.write_path, "w") as f:
            f.write("")

    def step(self, action):
        map_n = self.env.unwrapped.read_m("wCurMap")
        y_pos = self.env.unwrapped.read_m("wYCoord")
        x_pos = self.env.unwrapped.read_m("wXCoord")
        if self.step_counter >= self.record_interval:
            self.coord_list.append([str(map_n), str(y_pos), str(x_pos)])
            self.step_counter = 0
        if len(self.coord_list) >= self.write_frequency:
            self.write()
            self.coord_list.clear()

        self.step_counter += 1

        return self.env.step(action)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def close(self):
        self.write()
        return self.env.close()

    def write(self):
        with open(
            self.write_path,
            "a",
        ) as f:
            f.writelines(",".join(coord) + "\n" for coord in self.coord_list)


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
        with open(self.write_path, "w") as f:
            f.write("")

    def step(self, action):
        self.action_list.append(action)
        if len(self.action_list) >= self.write_frequency:
            self.write()
            self.action_list.clear()

        return self.env.step(action)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def close(self):
        self.write()
        return self.env.close()

    def write(self):
        with open(
            self.write_path,
            "a",
        ) as f:
            f.writelines(str(action) + "\n" for action in self.action_list)
