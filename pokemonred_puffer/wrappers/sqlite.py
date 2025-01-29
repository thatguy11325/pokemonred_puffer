from os import PathLike
import multiprocessing
import os
import sqlite3
from typing import Any

import gymnasium as gym

from pokemonred_puffer.environment import RedGymEnv


class SqliteStateResetWrapper(gym.Wrapper):
    DB_LOCK = multiprocessing.Lock()

    def __init__(
        self,
        env: RedGymEnv,
        database: str | bytes | PathLike[str] | PathLike[bytes],
    ):
        super().__init__(env)
        self.database = database
        with SqliteStateResetWrapper.DB_LOCK:
            with sqlite3.connect(database) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO states(env_id, pyboy_state, reset, required_rate, pid)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        self.env.unwrapped.env_id,
                        b"",
                        0,
                        1.0,
                        os.getpid(),
                    ),
                )
        print(f"Initialized sqlite row {self.env.unwrapped.env_id}")

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        with SqliteStateResetWrapper.DB_LOCK:
            with sqlite3.connect(self.database) as conn:
                cur = conn.cursor()
                reset, pyboy_state, required_rate = cur.execute(
                    """
                    SELECT reset, pyboy_state, required_rate
                    FROM states
                    WHERE env_id = ?
                    """,
                    (self.env.unwrapped.env_id,),
                ).fetchone()
                if reset:
                    if options:
                        options["state"] = pyboy_state
                    else:
                        options = {"state": pyboy_state}
                    cur.execute(
                        """
                        UPDATE states
                        SET reset = 0 
                        WHERE env_id = ?
                        """,
                        (self.env.unwrapped.env_id,),
                    )
        self.env.unwrapped.required_rate = required_rate
        return self.env.reset(seed=seed, options=options)
