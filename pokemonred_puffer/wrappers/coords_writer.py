import collections
import os
from datetime import datetime
from typing import cast

import gymnasium as gym
from omegaconf import DictConfig

from pokemonred_puffer.environment import RedGymEnv


class CoordinatesWriter(gym.Wrapper):
    def __init__(self, env: RedGymEnv, config: DictConfig):
        super().__init__(env)
        self.coord_list = collections.deque()
        self.output_dir: str = config.output_dir
        self.write_frequency: int = config.write_frequency
        self.folder_name = datetime.today.strftime("%Y%m%d%H%M%S")

    def step(self, action):
        map_n = self.env.unwrapped.read_m("wCurMap")
        y_pos = self.env.unwrapped.read_m("wYCoord")
        x_pos = self.env.unwrapped.read_m("wXCoord")
        self.coord_list.append((map_n, y_pos, x_pos))
        if len(self.coord_list) >= self.write_frequency:
            with open(
                os.path.join(self.output_dir, self.folder_name, cast(RedGymEnv, self.env).env_id),
                "a",
            ) as f:
                f.writelines(",".join(coord) + "\n" for coord in self.coord_list)
                self.coord_list.clear()

        return self.env.step(action)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
