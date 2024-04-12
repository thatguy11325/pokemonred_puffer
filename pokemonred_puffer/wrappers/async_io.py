import threading
from multiprocessing import Queue
import gymnasium as gym

from pokemonred_puffer.environment import RedGymEnv


class AsyncWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, send_queues: list[Queue], recv_queues: list[Queue]):
        super().__init__(env)
        self.send_queue = send_queues[self.env.unwrapped.env_id]
        self.recv_queue = recv_queues[self.env.unwrapped.env_id]
        # Now we will spawn a thread that will listen for updates
        # and send back when the new state has been loaded
        # this is a slow process and should rarely happen.
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        # TODO: Figure out if there's a safe way to exit the thread

    def update(self):
        while new_state := self.recv_queue.get():
            self.env.update_state(new_state)
            self.send_queue.put(self.env.unwrapped.env_id)
