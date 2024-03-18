import numpy as np
import pufferlib
from pokemonred_puffer.environment import RedGymEnv

EVENT_FLAGS_START_ADDR = 0xD747
EVENT_FLAGS_END_ADDR = 0xD886 # 0xD761
EVENTS_FLAGS_LENGTH = EVENT_FLAGS_END_ADDR - EVENT_FLAGS_START_ADDR
MUSEUM_TICKET_ADDR = 0xD754
USED_CELL_SEPARATOR_ADDR = 0xD7F2
MUSEUM_TICKET = (0xD754, 0)

class TeachCutEnv(RedGymEnv):
    def __init__(self, env_config: pufferlib.namespace, reward_config: pufferlib.namespace):
        super().__init__(env_config)

        self.reward_weights = reward_config if reward_config else {
            "event": 1.0,
            "bill_saved_reward": 5.0,
            "seen_pokemon_reward": 4.0,
            "caught_pokemon_reward": 4.0,
            "moves_obtained_reward": 4.0,
            "hm_count_reward": 10.0,
            "level": 1.0,
            "badges": 10.0,
            "exploration": 0.02,
            "used_cut_on_tree": 10.0,
            "cut_coords": 1.0,
            "cut_tiles": 1.0,
            "start_menu": 0.01,
            "pokemon_menu": 0.1,
            "stats_menu": 0.1,
            "bag_menu": 0.1
        }

    def get_game_state_reward(self):
        rewards = {
            "seen_pokemon_reward": self.reward_weights['seen_pokemon_reward'] * sum(self.seen_pokemon),
            "event": self.reward_weights['event'] * self.update_max_event_rew(),
            "exploration": self.reward_weights['exploration'] * sum(self.seen_coords.values()),
            "moves_obtained_reward": self.reward_weights['moves_obtained_reward'] * sum(self.moves_obtained),
            "level": self.reward_weights['level'] * self.get_levels_reward(),
            "caught_pokemon_reward": self.reward_weights['caught_pokemon_reward'] * sum(self.caught_pokemon),
            "badges": self.reward_weights['badges'] * self.get_badges(),
            "hm_count_reward": self.reward_weights['hm_count_reward'] * self.get_hm_count(),
            "bill_saved_reward": self.reward_weights['bill_saved_reward'] * self.saved_bill(),
            "cut_tiles": self.reward_weights['cut_tiles'] * len(self.cut_tiles),
            "start_menu": self.reward_weights['start_menu'] * self.seen_start_menu,
            "pokemon_menu": self.reward_weights['pokemon_menu'] * self.seen_pokemon_menu,
            "stats_menu": self.reward_weights['stats_menu'] * self.seen_stats_menu,
            "bag_menu": self.reward_weights['bag_menu'] * self.seen_bag_menu,
            # Add other rewards as needed
        }
        return rewards
    
    def seed(self, seed=None):
        # Implement seeding logic here if necessary
        pass

    def step(self, action):
        return super().step(action)

    def events(self):
        """Adds up all event flags, exclude museum ticket"""
        num_events = sum(
            self.bit_count(self.pyboy.get_memory_value(i))
            for i in range(EVENT_FLAGS_START_ADDR, EVENT_FLAGS_END_ADDR)
        )
        museum_ticket = int(self.read_bit(MUSEUM_TICKET_ADDR, 0))
        # Omit 13 events by default
        return max(num_events - 13 - museum_ticket, 0)

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew  
    
    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(EVENT_FLAGS_START_ADDR, EVENT_FLAGS_START_ADDR + EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(self.read_bit(*MUSEUM_TICKET)),
            0,
        )     

    def get_levels_reward(self):
            # Level reward
        party_levels = self.read_party()
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 30:
            level_reward = 1 * self.max_level_sum
        else:
            level_reward = 30 + (self.max_level_sum - 30) / 4
        return level_reward

    def get_items_in_bag(self, one_indexed=0):
        first_item = 0xD31E
        item_ids = []
        for i in range(0, 20, 2):
            item_id = self.pyboy.get_memory_value(first_item + i)
            if item_id == 0 or item_id == 0xff:
                break
            item_ids.append(item_id + one_indexed)
        return item_ids

    def get_hm_count(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.get_items_in_bag()
        total_hm_cnt = 0
        for hm_id in hm_ids:
            if hm_id in items:
                total_hm_cnt += 1
        return total_hm_cnt * 1

    def saved_bill(self):
        """Restored Bill from his experiment"""
        return int(self.read_bit(USED_CELL_SEPARATOR_ADDR, 3))
