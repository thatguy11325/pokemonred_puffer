import threading
from multiprocessing import Queue
import gymnasium as gym

from pokemonred_puffer.environment import RedGymEnv


class AsyncWrapper(gym.Wrapper):
    def __init__(self, env: RedGymEnv, send_queues: list[Queue], recv_queues: list[Queue]):
        super().__init__(env)
        # We need to -1 because the env id is one offset due to puffer's driver env
        self.send_queue = send_queues[self.env.unwrapped.env_id]
        self.recv_queue = recv_queues[self.env.unwrapped.env_id]
        print(f"Initialized queues for {self.env.unwrapped.env_id}")
        # Now we will spawn a thread that will listen for updates
        # and send back when the new state has been loaded
        # this is a slow process and should rarely happen.
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        # TODO: Figure out if there's a safe way to exit the thread

    def update(self):
        while True:
            new_state = self.recv_queue.get()
            if new_state == b"":
                print(f"invalid state for {self.env.unwrapped.env_id} skipping...")
            else:
                self.env.unwrapped.update_state(new_state)
            self.send_queue.put(self.env.unwrapped.env_id)
