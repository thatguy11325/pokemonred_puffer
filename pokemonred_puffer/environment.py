from abc import abstractmethod
import random
from collections import deque
from multiprocessing import Lock, shared_memory
from pathlib import Path
from typing import Optional
import uuid

import mediapy as media
import numpy as np
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from skimage.transform import resize

import pufferlib
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE, local_to_global

EVENT_FLAGS_START = 0xD747
EVENTS_FLAGS_LENGTH = 320
MUSEUM_TICKET = (0xD754, 0)
PARTY_SIZE = 0xD163
PARTY_LEVEL_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

CUT_SEQ = [
    ((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)),
    ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),
]

CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])

VISITED_MASK_SHAPE = (144 // 16, 160 // 16, 1)

TM_HM_MOVES = set(
    [
        5,  # Mega punch
        0xD,  # Razor wind
        0xE,  # Swords dance
        0x12,  # Whirlwind
        0x19,  # Mega kick
        0x5C,  # Toxic
        0x20,  # Horn drill
        0x22,  # Body slam
        0x24,  # Take down
        0x26,  # Double edge
        0x3D,  # Bubble beam
        0x37,  # Water gun
        0x3A,  # Ice beam
        0x3B,  # Blizzard
        0x3F,  # Hyper beam
        0x06,  # Pay day
        0x42,  # Submission
        0x44,  # Counter
        0x45,  # Seismic toss
        0x63,  # Rage
        0x48,  # Mega drain
        0x4C,  # Solar beam
        0x52,  # Dragon rage
        0x55,  # Thunderbolt
        0x57,  # Thunder
        0x59,  # Earthquake
        0x5A,  # Fissure
        0x5B,  # Dig
        0x5E,  # Psychic
        0x64,  # Teleport
        0x66,  # Mimic
        0x68,  # Double team
        0x73,  # Reflect
        0x75,  # Bide
        0x76,  # Metronome
        0x78,  # Selfdestruct
        0x79,  # Egg bomb
        0x7E,  # Fire blast
        0x81,  # Swift
        0x82,  # Skull bash
        0x87,  # Softboiled
        0x8A,  # Dream eater
        0x8F,  # Sky attack
        0x9C,  # Rest
        0x56,  # Thunder wave
        0x95,  # Psywave
        0x99,  # Explosion
        0x9D,  # Rock slide
        0xA1,  # Tri attack
        0xA4,  # Substitute
        0x0F,  # Cut
        0x13,  # Fly
        0x39,  # Surf
        0x46,  # Strength
        0x94,  # Flash
    ]
)

RESET_MAP_IDS = set(
    [
        0x0,  # Pallet Town
        0x1,  # Viridian City
        0x2,  # Pewter City
        0x3,  # Cerulean City
        0x4,  # Lavender Town
        0x5,  # Vermilion City
        0x6,  # Celadon City
        0x7,  # Fuchsia City
        0x8,  # Cinnabar Island
        0x9,  # Indigo Plateau
        0xA,  # Saffron City
        0xF,  # Route 4 (Mt Moon)
        0x10,  # Route 10 (Rock Tunnel)
        0xE9,  # Silph Co 9F (Heal station)
    ]
)

VALID_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

ACTION_SPACE = spaces.Discrete(len(VALID_ACTIONS))


# TODO: Make global map usage a configuration parameter
class RedGymEnv(Env):
    env_id = shared_memory.SharedMemory(create=True, size=4)
    lock = Lock()

    def __init__(self, env_config: pufferlib.namespace):
        # TODO: Dont use pufferlib.namespace. It seems to confuse __init__
        self.video_dir = Path(env_config.video_dir)
        self.session_path = Path(env_config.session_path)
        self.video_path = self.video_dir / self.session_path
        self.save_final_state = env_config.save_final_state
        self.print_rewards = env_config.print_rewards
        self.headless = env_config.headless
        self.state_dir = Path(env_config.state_dir)
        self.init_state = env_config.init_state
        self.init_state_name = self.init_state
        self.init_state_path = self.state_dir / f"{self.init_state_name}.state"
        self.action_freq = env_config.action_freq
        self.max_steps = env_config.max_steps
        self.save_video = env_config.save_video
        self.fast_video = env_config.fast_video
        self.frame_stacks = env_config.frame_stacks
        self.perfect_ivs = env_config.perfect_ivs
        self.reduce_res = env_config.reduce_res
        self.gb_path = env_config.gb_path
        self.action_space = ACTION_SPACE

        # Obs space-related. TODO: avoid hardcoding?
        if self.reduce_res:
            self.screen_output_shape = (72, 80, 3 * self.frame_stacks)
        else:
            self.screen_output_shape = (144, 160, 3 * self.frame_stacks)
        self.coords_pad = 12
        self.enc_freqs = 8

        # NOTE: Used for saving video
        if env_config.save_video:
            self.instance_id = str(uuid.uuid4())[:8]
            self.video_dir.mkdir(exist_ok=True)
            self.full_frame_writer = None
            self.model_frame_writer = None
            self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.essential_map_locations = {
            v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        }

        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(
                    low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
                ),
                # Discrete is more apt, but pufferlib is slower at processing Discrete
                "direction": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
                "reset_map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
                "battle_type": spaces.Box(low=0, high=4, shape=(1,), dtype=np.uint8),
                "cut_in_party": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
                "x": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                "y": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                "map_id": spaces.Box(low=0, high=0xF7, shape=(1,), dtype=np.uint8),
                "badges": spaces.Box(low=0, high=8, shape=(1,), dtype=np.uint8),
            }
        )

        self.pyboy = PyBoy(
            env_config.gb_path,
            debugging=False,
            disable_input=False,
            window_type="headless" if self.headless else "SDL2",
        )
        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        self.screen = self.pyboy.botsupport_manager().screen()

        self.first = True
        with RedGymEnv.lock:
            env_id = (
                (int(RedGymEnv.env_id.buf[0]) << 24)
                + (int(RedGymEnv.env_id.buf[1]) << 16)
                + (int(RedGymEnv.env_id.buf[2]) << 8)
                + (int(RedGymEnv.env_id.buf[3]))
            ) + 1
            self.env_id = env_id
            RedGymEnv.env_id.buf[0] = (env_id >> 24) & 0xFF
            RedGymEnv.env_id.buf[1] = (env_id >> 16) & 0xFF
            RedGymEnv.env_id.buf[2] = (env_id >> 8) & 0xFF
            RedGymEnv.env_id.buf[3] = (env_id) & 0xFF

    def reset(self, seed: Optional[int] = None):
        # restart game, skipping credits
        self.explore_map_dim = 384
        if self.first:
            self.recent_screens = deque()  # np.zeros(self.output_shape, dtype=np.uint8)
            self.recent_actions = deque()  # np.zeros((self.frame_stacks,), dtype=np.uint8)
            self.seen_pokemon = np.zeros(152, dtype=np.uint8)
            self.caught_pokemon = np.zeros(152, dtype=np.uint8)
            self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
            self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
            self.init_mem()
            self.reset_count = 0
        else:
            self.recent_screens.clear()
            self.recent_actions.clear()
            self.seen_pokemon.fill(0)
            self.caught_pokemon.fill(0)
            self.moves_obtained.fill(0)
            self.explore_map *= 0
            self.reset_mem()
            self.reset_count += 1

        with open(self.init_state_path, "rb") as f:
            self.pyboy.load_state(f)

        # lazy random seed setting
        if not seed:
            seed = random.randint(0, 4096)
        for _ in range(seed):
            self.pyboy.tick()

        self.taught_cut = self.check_if_party_has_cut()
        self.base_event_flags = sum(
            self.bit_count(self.read_m(i))
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
        )

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.max_level_sum = 0
        self.last_health = 1
        self.total_heal_health = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.blackout_check = 0
        self.blackout_count = 0
        self.blackout_debounce = True

        self.current_event_flags_set = {}

        self.action_hist = np.zeros(len(VALID_ACTIONS))

        # experiment!
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

        self.first = False
        return self._get_obs(), {}

    def init_mem(self):
        # Maybe I should preallocate a giant matrix for all map ids
        # All map ids have the same size, right?
        self.seen_coords = {}
        self.seen_coords_since_blackout = set([])
        # self.seen_global_coords = np.zeros(GLOBAL_MAP_SHAPE)
        self.seen_map_ids = np.zeros(256)
        self.seen_map_ids_since_blackout = set([])

        self.seen_npcs = {}
        self.seen_npcs_since_blackout = set([])

        self.seen_hidden_objs = {}

        self.cut_coords = {}
        self.cut_tiles = set([])
        self.cut_state = deque(maxlen=3)

        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0

    def reset_mem(self):
        self.seen_coords.update((k, 0) for k, _ in self.seen_coords.items())
        self.seen_map_ids *= 0

        self.seen_npcs.update((k, 0) for k, _ in self.seen_npcs.items())

        self.seen_hidden_objs.update((k, 0) for k, _ in self.seen_hidden_objs.items())

        self.cut_coords.update((k, 0) for k, _ in self.cut_coords.items())
        self.cut_state = deque(maxlen=3)

        self.seen_start_menu = 0
        self.seen_pokemon_menu = 0
        self.seen_stats_menu = 0
        self.seen_bag_menu = 0
        self.seen_action_bag_menu = 0

    def blackout(self):
        if self.blackout_debounce and self.read_m(0xCF0B) == 0x01:
            self.blackout_count += 1
            self.blackout_debounce = False
        elif self.read_m(0xD057) == 0:
            self.blackout_debounce = True

    def render(self):
        # (144, 160, 3)
        game_pixels_render = self.screen.screen_ndarray()[:, :, 0:1]
        if self.reduce_res:
            game_pixels_render = game_pixels_render[::2, ::2, :]
        # place an overlay on top of the screen greying out places we haven't visited
        # first get our location
        player_x, player_y, map_n = self.get_game_coords()

        # player is centered at 68, 72 in pixel units
        # 68 -> player y, 72 -> player x
        # guess we want to attempt to map the pixels to player units or vice versa
        # Experimentally determined magic numbers below. Beware
        # visited_mask = np.zeros(VISITED_MASK_SHAPE, dtype=np.float32)
        visited_mask = np.zeros_like(game_pixels_render)
        """
        if self.taught_cut:
            cut_mask = np.zeros_like(game_pixels_render)
        else:
            cut_mask = np.random.randint(0, 255, game_pixels_render.shape, dtype=np.uint8)
        """
        # If not in battle, set the visited mask. There's no reason to process it when in battle
        scale = 2 if self.reduce_res else 1
        if self.read_m(0xD057) == 0:
            for y in range(-72 // 16, 72 // 16):
                for x in range(-80 // 16, 80 // 16):
                    # y-y1 = m (x-x1)
                    # map [(0,0),(1,1)] -> [(0,.5),(1,1)] (cause we dont wnat it to be fully black)
                    # y = 1/2 x + .5
                    # current location tiles - player_y*8, player_x*8
                    """
                    visited_mask[y, x, 0] = self.seen_coords.get(
                        (
                            player_x + x + 1,
                            player_y + y + 1,
                            map_n,
                        ),
                        0.15,
                    )
                    """

                    visited_mask[
                        (16 * y + 76) // scale : (16 * y + 16 + 76) // scale,
                        (16 * x + 80) // scale : (16 * x + 16 + 80) // scale,
                        :,
                    ] = int(
                        self.seen_coords.get(
                            (
                                player_x + x + 1,
                                player_y + y + 1,
                                map_n,
                            ),
                            0.15,
                        )
                        * 255
                    )
                    """
                    if self.taught_cut:
                        cut_mask[
                            16 * y + 76 : 16 * y + 16 + 76,
                            16 * x + 80 : 16 * x + 16 + 80,
                            :,
                        ] = int(
                            255
                            * (
                                self.cut_coords.get(
                                    (
                                        player_x + x + 1,
                                        player_y + y + 1,
                                        map_n,
                                    ),
                                    0,
                                )
                            )
                        )
                        """
        """
        gr, gc = local_to_global(player_y, player_x, map_n)
        visited_mask = (
            255
            * np.repeat(
                np.repeat(self.seen_global_coords[gr - 4 : gr + 5, gc - 4 : gc + 6], 16, 0), 16, -1
            )
        ).astype(np.uint8)
        visited_mask = np.expand_dims(visited_mask, -1)
        """
        # game_pixels_render = np.concatenate([game_pixels_render, visited_mask, cut_mask], axis=-1)
        game_pixels_render = np.concatenate([game_pixels_render, visited_mask], axis=-1)

        return game_pixels_render

    def _get_screen_obs(self):
        screen = self.render()
        screen = np.concatenate(
            [
                screen,
                np.expand_dims(
                    255 * resize(self.explore_map, screen.shape[:-1], anti_aliasing=False),
                    axis=-1,
                ).astype(np.uint8),
            ],
            axis=-1,
        )

        self.update_recent_screens(screen)
        return screen

    def _get_obs(self):
        player_x, player_y, map_n = self.get_game_coords()
        return {
            "screen": self._get_screen_obs(),
            "direction": np.array(self.pyboy.get_memory_value(0xC109) // 4, dtype=np.uint8),
            "reset_map_id": np.array(self.pyboy.get_memory_value(0xD719), dtype=np.uint8),
            "battle_type": np.array(self.pyboy.get_memory_value(0xD057) + 1, dtype=np.uint8),
            "cut_in_party": np.array(self.check_if_party_has_cut(), dtype=np.uint8),
            "x": np.array(player_x, dtype=np.uint8),
            "y": np.array(player_y, dtype=np.uint8),
            "map_id": np.array(map_n, dtype=np.uint8),
            "badges": np.array(self.get_badges(), dtype=np.uint8),
        }

    def set_perfect_iv_dvs(self):
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(12):  # Number of offsets for IV/DV
                self.pyboy.set_memory_value(i + 17 + m, 0xFF)

    def check_if_party_has_cut(self) -> bool:
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(4):
                if self.pyboy.get_memory_value(i + 8 + m) == 15:
                    return True
        return False

    def check_if_in_start_menu(self) -> bool:
        return (
            self.read_m(0xD057) == 0
            and self.read_m(0xCF13) == 0
            and self.read_m(0xFF8C) == 6
            and self.read_m(0xCF94) == 0
        )

    def check_if_in_pokemon_menu(self) -> bool:
        return (
            self.read_m(0xD057) == 0
            and self.read_m(0xCF13) == 0
            and self.read_m(0xFF8C) == 6
            and self.read_m(0xCF94) == 2
        )

    def check_if_in_stats_menu(self) -> bool:
        return (
            self.read_m(0xD057) == 0
            and self.read_m(0xCF13) == 0
            and self.read_m(0xFF8C) == 6
            and self.read_m(0xCF94) == 1
        )

    def check_if_in_bag_menu(self) -> bool:
        return (
            self.read_m(0xD057) == 0
            and self.read_m(0xCF13) == 0
            # and self.read_m(0xFF8C) == 6 # only sometimes
            and self.read_m(0xCF94) == 3
        )

    def check_if_action_in_bag_menu(self, action) -> bool:
        return action == WindowEvent.PRESS_BUTTON_A and self.check_if_in_bag_menu()

    def check_if_in_overworld(self) -> bool:
        return self.read_m(0xD057) == 0 and self.read_m(0xCF13) == 0 and self.read_m(0xFF8C) == 0

    def update_blackout(self):
        cur_map_id = self.read_m(0xD35E)
        if cur_map_id in RESET_MAP_IDS:
            blackout_check = int(cur_map_id == self.read_m(0xD719))
            if blackout_check and not self.blackout_check:
                self.seen_coords_since_blackout.clear()
                self.seen_npcs_since_blackout.clear()
                self.seen_map_ids_since_blackout.clear()

                self.blackout_check = blackout_check

    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.update_seen_coords()
        self.update_health()
        self.update_pokedex()
        self.update_tm_hm_moves_obtained()
        self.party_size = self.read_m(0xD163)
        self.update_max_op_level()
        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        if self.perfect_ivs:
            self.set_perfect_iv_dvs()
        self.taught_cut = self.check_if_party_has_cut()
        self.blackout()
        self.update_blackout()

        info = {}
        # TODO: Make log frequency a configuration parameter
        if self.step_count % 2000 == 0:
            info = self.agent_stats(action)

        obs = self._get_obs()

        self.step_count += 1
        reset = (
            self.step_count > self.max_steps  # or
            # self.caught_pokemon[6] == 1  # squirtle
        )

        return obs, new_reward, reset, False, info

    def find_neighboring_sign(self, sign_id, player_direction, player_x, player_y) -> bool:
        sign_y = self.pyboy.get_memory_value(0xD4B1 + (2 * sign_id))
        sign_x = self.pyboy.get_memory_value(0xD4B1 + (2 * sign_id + 1))

        # Check if player is facing the sign (skip sign direction)
        # 0 - down, 4 - up, 8 - left, 0xC - right
        # We are making the assumption that a player will only ever be 1 space away
        # from a sign
        return (
            (player_direction == 0 and sign_x == player_x and sign_y == player_y + 1)
            or (player_direction == 4 and sign_x == player_x and sign_y == player_y - 1)
            or (player_direction == 8 and sign_y == player_y and sign_x == player_x - 1)
            or (player_direction == 0xC and sign_y == player_y and sign_x == player_x + 1)
        )

    def find_neighboring_npc(self, npc_id, player_direction, player_x, player_y) -> int:
        npc_y = self.pyboy.get_memory_value(0xC104 + (npc_id * 0x10))
        npc_x = self.pyboy.get_memory_value(0xC106 + (npc_id * 0x10))

        # Check if player is facing the NPC (skip NPC direction)
        # 0 - down, 4 - up, 8 - left, 0xC - right
        if (
            (player_direction == 0 and npc_x == player_x and npc_y > player_y)
            or (player_direction == 4 and npc_x == player_x and npc_y < player_y)
            or (player_direction == 8 and npc_y == player_y and npc_x < player_x)
            or (player_direction == 0xC and npc_y == player_y and npc_x > player_x)
        ):
            # Manhattan distance
            return abs(npc_y - player_y) + abs(npc_x - player_x)

        return False

    def run_action_on_emulator(self, action):
        self.action_hist[action] += 1

        # press button then release after some steps
        self.pyboy.send_input(VALID_ACTIONS[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.action_freq):
            # release action, so they are stateless
            if i == 8 and action < len(RELEASE_ACTIONS):
                # release button
                self.pyboy.send_input(RELEASE_ACTIONS[action])

            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.action_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()

        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        if self.read_m(0xD057) == 0:
            if self.taught_cut:
                player_direction = self.pyboy.get_memory_value(0xC109)
                x, y, map_id = self.get_game_coords()  # x, y, map_id
                if player_direction == 0:  # down
                    coords = (x, y + 1, map_id)
                if player_direction == 4:
                    coords = (x, y - 1, map_id)
                if player_direction == 8:
                    coords = (x - 1, y, map_id)
                if player_direction == 0xC:
                    coords = (x + 1, y, map_id)
                self.cut_state.append(
                    (
                        self.pyboy.get_memory_value(0xCFC6),
                        self.pyboy.get_memory_value(0xCFCB),
                        self.pyboy.get_memory_value(0xCD6A),
                        self.pyboy.get_memory_value(0xD367),
                        self.pyboy.get_memory_value(0xD125),
                        self.pyboy.get_memory_value(0xCD3D),
                    )
                )
                if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                    self.cut_coords[coords] = 10
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif self.cut_state == CUT_GRASS_SEQ:
                    self.cut_coords[coords] = 0.01
                    self.cut_tiles[self.cut_state[-1][0]] = 1
                elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
                    self.cut_coords[coords] = 0.01
                    self.cut_tiles[self.cut_state[-1][0]] = 1

            # check if the font is loaded
            if self.pyboy.get_memory_value(0xCFC4):
                # check if we are talking to a hidden object:
                player_direction = self.pyboy.get_memory_value(0xC109)
                player_y_tiles = self.pyboy.get_memory_value(0xD361)
                player_x_tiles = self.pyboy.get_memory_value(0xD362)
                if (
                    self.pyboy.get_memory_value(0xCD3D) != 0x0
                    and self.pyboy.get_memory_value(0xCD3E) != 0x0
                ):
                    # add hidden object to seen hidden objects
                    self.seen_hidden_objs[
                        (
                            self.pyboy.get_memory_value(0xD35E),
                            self.pyboy.get_memory_value(0xCD3F),
                        )
                    ] = 1
                elif any(
                    self.find_neighboring_sign(
                        sign_id, player_direction, player_x_tiles, player_y_tiles
                    )
                    for sign_id in range(self.pyboy.get_memory_value(0xD4B0))
                ):
                    pass
                else:
                    # get information for player
                    player_y = self.pyboy.get_memory_value(0xC104)
                    player_x = self.pyboy.get_memory_value(0xC106)
                    # get the npc who is closest to the player and facing them
                    # we go through all npcs because there are npcs like
                    # nurse joy who can be across a desk and still talk to you

                    # npc_id 0 is the player
                    npc_distances = (
                        (
                            self.find_neighboring_npc(npc_id, player_direction, player_x, player_y),
                            npc_id,
                        )
                        for npc_id in range(1, self.pyboy.get_memory_value(0xD4E1))
                    )
                    npc_candidates = [x for x in npc_distances if x[0]]
                    if npc_candidates:
                        _, npc_id = min(npc_candidates, key=lambda x: x[0])
                        self.seen_npcs[(self.pyboy.get_memory_value(0xD35E), npc_id)] = 1
                        self.seen_npcs_since_blackout.add(
                            (self.pyboy.get_memory_value(0xD35E), npc_id)
                        )

                if int(self.read_bit(0xD803, 0)):
                    if self.check_if_in_start_menu():
                        self.seen_start_menu = 1

                    if self.check_if_in_pokemon_menu():
                        self.seen_pokemon_menu = 1

                    if self.check_if_in_stats_menu():
                        self.seen_stats_menu = 1

                    if self.check_if_in_bag_menu():
                        self.seen_bag_menu = 1

                    if self.check_if_action_in_bag_menu(action):
                        self.seen_action_bag_menu = 1

        if self.save_video and self.fast_video:
            self.add_video_frame()

    def agent_stats(self, action):
        levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return {
            "stats": {
                "step": self.step_count + self.reset_count * self.max_steps,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord": sum(self.seen_coords.values()),  # np.sum(self.seen_global_coords),
                "map_id": np.sum(self.seen_map_ids),
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_heal_health,
                "action_hist": self.action_hist,
                "caught_pokemon": int(sum(self.caught_pokemon)),
                "seen_pokemon": int(sum(self.seen_pokemon)),
                "moves_obtained": int(sum(self.moves_obtained)),
                "opponent_level": self.max_opponent_level,
                "met_bill": int(self.read_bit(0xD7F1, 0)),
                "used_cell_separator_on_bill": int(self.read_bit(0xD7F2, 3)),
                "ss_ticket": int(self.read_bit(0xD7F2, 4)),
                "met_bill_2": int(self.read_bit(0xD7F2, 5)),
                "bill_said_use_cell_separator": int(self.read_bit(0xD7F2, 6)),
                "left_bills_house_after_helping": int(self.read_bit(0xD7F2, 7)),
                "got_hm01": int(self.read_bit(0xD803, 0)),
                "rubbed_captains_back": int(self.read_bit(0xD803, 1)),
                "taught_cut": int(self.check_if_party_has_cut()),
                "cut_coords": sum(self.cut_coords.values()),
                "cut_tiles": len(self.cut_tiles),
                "start_menu": self.seen_start_menu,
                "pokemon_menu": self.seen_pokemon_menu,
                "stats_menu": self.seen_stats_menu,
                "bag_menu": self.seen_bag_menu,
                "action_bag_menu": self.seen_action_bag_menu,
                "blackout_check": self.blackout_check,
                "item_count": self.read_m(0xD31D),
                "reset_count": self.reset_count,
                "blackout_count": self.blackout_count,
            },
            "reward": self.get_game_state_reward(),
            "reward/reward_sum": sum(self.get_game_state_reward().values()),
            "pokemon_exploration_map": self.explore_map,
        }

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f"full_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        model_name = Path(f"model_reset_{self.reset_count}_id{self.instance_id}").with_suffix(
            ".mp4"
        )
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.screen_output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(f"map_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render()[:, :, 0])
        self.model_frame_writer.add_image(self.render()[:, :, 0])

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        self.seen_coords[(x_pos, y_pos, map_n)] = 1
        self.seen_coords_since_blackout.add((x_pos, y_pos, map_n))
        self.explore_map[local_to_global(y_pos, x_pos, map_n)] = 1
        # self.seen_global_coords[local_to_global(y_pos, x_pos, map_n)] = 1
        self.seen_map_ids[map_n] = 1
        self.seen_map_ids_since_blackout.add(map_n)

    def get_explore_map(self):
        explore_map = np.zeros(GLOBAL_MAP_SHAPE)
        for (x, y, map_n), v in self.seen_coords.items():
            gy, gx = local_to_global(y, x, map_n)
            if gy >= explore_map.shape[0] or gy < 0 or gx >= explore_map.shape[1] or gx < 0:
                print(f"coord out of bounds! global: ({gx}, {gy}) game: ({x}, {y}, {map_n})")
            else:
                explore_map[gy, gx] = v

        return explore_map

    def update_recent_screens(self, cur_screen):
        # self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        # self.recent_screens[:, :, 0] = cur_screen[:, :, 0]
        self.recent_screens.append(cur_screen)
        if len(self.recent_screens) > self.frame_stacks:
            self.recent_screens.popleft()

    def update_recent_actions(self, action):
        # self.recent_actions = np.roll(self.recent_actions, 1)
        # self.recent_actions[0] = action
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.frame_stacks:
            self.recent_actions.popleft()

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit)
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]

    @abstractmethod
    def get_game_state_reward(self):
        raise NotImplementedError()

    def update_max_op_level(self):
        # opp_base_level = 5
        opponent_level = (
            max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]])
            # - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_health(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                self.total_heal_health += cur_health - self.last_health
            else:
                self.died_count += 1

    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.pyboy.get_memory_value(i + 0xD2F7)
            seen_mem = self.pyboy.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8 * i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8 * i + j] = 1 if seen_mem & (1 << j) else 0

    def update_tm_hm_moves_obtained(self):
        # Scan party
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
            if self.pyboy.get_memory_value(i) != 0:
                for j in range(4):
                    move_id = self.pyboy.get_memory_value(i + j + 8)
                    if move_id != 0:  # and move_id in TM_HM_MOVES:
                        self.moves_obtained[move_id] = 1
        """
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.pyboy.get_memory_value(0xDA80)):
            offset = i * box_struct_length + 0xDA96
            if self.pyboy.get_memory_value(offset) != 0:
                for j in range(4):
                    move_id = self.pyboy.get_memory_value(offset + j + 8)
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1
        """

    def read_hp_fraction(self):
        hp_sum = sum(
            [self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]]
        )
        max_hp_sum = sum(
            [self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]]
        )
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    def bit_count(self, bits):
        return bits.bit_count()

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1
