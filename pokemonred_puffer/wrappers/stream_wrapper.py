import asyncio
import json
import os

import gymnasium as gym
import websockets
from omegaconf import DictConfig

from pokemonred_puffer.environment import RedGymEnv


class StreamWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, config: DictConfig):
        super().__init__(env)

        self.user = config.user
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = {"user": self.user, "env_id": self.env_id, "extra": os.getpid()}
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = self.loop.run_until_complete(self.establish_wc_connection())
        self.upload_interval = 500
        self.steam_step_counter = 0
        self.coord_list = []
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

    def step(self, action):
        x_pos = self.env.unwrapped.read_m("wXCoord")
        y_pos = self.env.unwrapped.read_m("wYCoord")
        map_n = self.env.unwrapped.read_m("wCurMap")
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.steam_step_counter >= self.upload_interval:
            self.loop.run_until_complete(
                self.broadcast_ws_message(
                    json.dumps({"metadata": self.stream_metadata, "coords": self.coord_list})
                )
            )
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        return self.env.step(action)

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except:  # noqa
            self.websocket = None

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
