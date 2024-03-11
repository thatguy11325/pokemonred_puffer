import gymnasium as gym
import numpy as np

import pufferlib
from pokemonred_puffer.environment import RedGymEnv


# Yes. This wrapper mutates the env.
# Is that good? No.
# Am I doing it anyway? Yes.
# Why? To save memory
class DecayWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, reward_config: pufferlib.namespace):
        super().__init__(env)
        self.step_forgetting_factor = reward_config.step_forgetting_factor
        self.forgetting_frequency = reward_config.forgetting_frequency

    def step(self, action):
        if self.env.unwrapped.step_count % self.forgetting_frequency == 0:
            self.step_forget_explore()

        return self.env.step(action)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step_forget_explore(self):
        self.env.unwrapped.seen_coords.update(
            (k, max(0.15, v * (self.step_forgetting_factor["coords"])))
            for k, v in self.env.unwrapped.seen_coords.items()
        )
        self.env.unwrapped.seen_map_ids *= self.step_forgetting_factor["map_ids"]
        self.env.unwrapped.seen_npcs.update(
            (k, max(0.15, v * (self.step_forgetting_factor["npc"])))
            for k, v in self.env.unwrapped.seen_npcs.items()
        )
        self.env.unwrapped.explore_map *= self.step_forgetting_factor["explore"]
        self.env.unwrapped.explore_map[self.env.unwrapped.explore_map > 0] = np.clip(
            self.env.unwrapped.explore_map[self.env.unwrapped.explore_map > 0], 0.15, 1
        )

        if self.env.unwrapped.read_m(0xD057) == 0:
            self.env.unwrapped.seen_start_menu *= self.step_forgetting_factor["start_menu"]
            self.env.unwrapped.seen_pokemon_menu *= self.step_forgetting_factor["pokemon_menu"]
            self.env.unwrapped.seen_stats_menu *= self.step_forgetting_factor["stats_menu"]
            self.env.unwrapped.seen_bag_menu *= self.step_forgetting_factor["bag_menu"]
            self.env.unwrapped.seen_action_bag_menu *= self.step_forgetting_factor[
                "action_bag_menu"
            ]
