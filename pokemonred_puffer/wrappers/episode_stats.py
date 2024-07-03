import numpy as np
import gymnasium

import pufferlib.utils


class EpisodeStatsWrapper(gymnasium.Wrapper):
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset()

    # TODO: Fix options. Maybe reimplement gymnasium.Wrapper with better compatibility
    def reset(self, seed=None):
        self.info = dict(episode_return=0, episode_length=0)
        return super().reset(seed=seed)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        for k, v in pufferlib.utils.unroll_nested_dict(info):
            if "exploration_map" in k:
                self.info[k] = self.info.get(k, np.zeros_like(v)) + v
            elif "state" in k:
                breakpoint()
                self.info["state"] |= v
            else:
                self.info[k] = v

        # self.info['episode_return'].append(reward)
        self.info["episode_return"] += reward
        self.info["episode_length"] += 1

        info = {}
        if terminated or truncated or self.info["episode_length"] % self.env.log_frequency == 0:
            info = self.info

        return observation, reward, terminated, truncated, info
