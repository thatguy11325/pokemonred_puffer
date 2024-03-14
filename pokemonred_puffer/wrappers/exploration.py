from collections import OrderedDict
import gymnasium as gym
import numpy as np

import pufferlib
from pokemonred_puffer.environment import RedGymEnv


class LRUCache:
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def contains(self, key: tuple[int, int, int]) -> bool:
        if key not in self.cache:
            return False
        else:
            self.cache.move_to_end(key)
            return True

    def put(self, key: tuple[int, int, int]) -> tuple[int, int, int] | None:
        self.cache[key] = 1
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            return self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


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


class MaxLengthWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, reward_config: pufferlib.namespace):
        super().__init__(env)
        self.capacity = reward_config.capacity
        self.cache = LRUCache(capacity=self.capacity)

    def step(self, action):
        player_x, player_y, map_n = self.env.unwrapped.get_game_coords()
        # Walrus operator does not support tuple unpacking
        if coord := self.cache.put((player_x, player_y, map_n)):
            x, y, n = coord
            del self.env.unwrapped.seen_coords[(x, y, n)]
            self.env.unwrapped.explore_map[self.env.unwrapped.local_to_global(y, x, n)] = 0

        return self.env.step(action)

    def reset(self, *args, **kwargs):
        self.cache.clear()
        return self.env.reset(*args, **kwargs)
