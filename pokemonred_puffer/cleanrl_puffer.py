import argparse
import ast
from functools import partial
import heapq
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from multiprocessing import Queue

import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.pytorch
import pufferlib.utils
import pufferlib.vector

# Fast Cython GAE implementation
import pyximport
import rich
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

import wandb
from pokemonred_puffer.eval import make_pokemon_red_overlay
from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE
from pokemonred_puffer.profile import Profile, Utilization

pyximport.install(setup_args={"include_dirs": np.get_include()})
from pokemonred_puffer.c_gae import compute_gae  # type: ignore  # noqa: E402


def rollout(
    env_creator,
    env_kwargs,
    agent_creator,
    agent_kwargs,
    model_path=None,
    device="cuda",
):
    # We are just using Serial vecenv to give a consistent
    # single-agent/multi-agent API for evaluation
    try:
        env = pufferlib.vector.make(
            env_creator, env_kwargs={"render_mode": "rgb_array", **env_kwargs}
        )
    except:  # noqa: E722
        env = pufferlib.vector.make(env_creator, env_kwargs=env_kwargs)

    if model_path is None:
        agent = agent_creator(env, **agent_kwargs)
    else:
        agent = torch.load(model_path, map_location=device)

    ob, info = env.reset()
    driver = env.driver_env
    os.system("clear")
    state = None

    while True:
        render = driver.render()
        if driver.render_mode == "ansi":
            print("\033[0;0H" + render + "\n")
            time.sleep(0.6)
        elif driver.render_mode == "rgb_array":
            import cv2

            render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", render)
            cv2.waitKey(1)
            time.sleep(1 / 24)

        with torch.no_grad():
            ob = torch.from_numpy(ob).to(device)
            if hasattr(agent, "lstm"):
                action, _, _, _, state = agent(ob, state)
            else:
                action, _, _, _ = agent(ob)

            action = action.cpu().numpy().reshape(env.action_space.shape)

        ob, reward = env.step(action)[:2]
        reward = reward.mean()
        print(f"Reward: {reward:.4f}")


def seed_everything(seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def unroll_nested_dict(d):
    if not isinstance(d, dict):
        return d

    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in unroll_nested_dict(v):
                yield f"{k}/{k2}", v2
        else:
            yield k, v


def count_params(policy: nn.Module):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


@dataclass
class Losses:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    old_approx_kl: float = 0.0
    approx_kl: float = 0.0
    clipfrac: float = 0.0
    explained_variance: float = 0.0


@dataclass
class CleanPuffeRL:
    exp_name: str
    config: argparse.Namespace
    vecenv: pufferlib.vector.Serial | pufferlib.vector.Multiprocessing
    policy: nn.Module
    env_send_queues: list[Queue]
    env_recv_queues: list[Queue]
    wandb_client: wandb.wandb_sdk.wandb_run.Run | None = None
    profile: Profile = field(default_factory=lambda: Profile())
    losses: Losses = field(default_factory=lambda: Losses())
    global_step: int = 0
    epoch: int = 0
    stats: dict = field(default_factory=lambda: {})
    msg: str = ""
    infos: dict = field(default_factory=lambda: defaultdict(list))
    states: dict = field(default_factory=lambda: defaultdict(partial(deque, maxlen=5)))
    event_tracker: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        seed_everything(self.config.seed, self.config.torch_deterministic)
        if self.config.verbose:
            self.utilization = Utilization()
            print_dashboard(
                self.config.env,
                self.utilization,
                0,
                0,
                self.profile,
                self.losses,
                {},
                self.msg,
                clear=True,
            )

        self.vecenv.async_reset(self.config.seed)
        obs_shape = self.vecenv.single_observation_space.shape
        obs_dtype = self.vecenv.single_observation_space.dtype
        atn_shape = self.vecenv.single_action_space.shape
        total_agents = self.vecenv.num_agents

        self.lstm = self.policy.lstm if hasattr(self.policy, "lstm") else None
        self.experience = Experience(
            self.config.batch_size,
            self.vecenv.agents_per_batch,
            self.config.bptt_horizon,
            self.config.minibatch_size,
            obs_shape,
            obs_dtype,
            atn_shape,
            self.config.cpu_offload,
            self.config.device,
            self.lstm,
            total_agents,
        )

        self.uncompiled_policy = self.policy

        if self.config.compile:
            self.policy = torch.compile(self.policy, mode=self.config.compile_mode)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

        self.last_log_time = time.time()

        self.reward_buffer = deque(maxlen=1_000)
        self.exploration_map_agg = np.zeros(
            (self.config.num_envs, *GLOBAL_MAP_SHAPE), dtype=np.float32
        )
        self.cut_exploration_map_agg = np.zeros(
            (self.config.num_envs, *GLOBAL_MAP_SHAPE), dtype=np.float32
        )
        self.taught_cut = False
        self.log = False

    @pufferlib.utils.profile
    def evaluate(self):
        # states are managed separately so dont worry about deleting them
        for k in list(self.infos.keys()):
            del self.infos[k]

        with self.profile.eval_misc:
            policy = self.policy
            lstm_h, lstm_c = self.experience.lstm_h, self.experience.lstm_c

        while not self.experience.full:
            with self.profile.env:
                o, r, d, t, info, env_id, mask = self.vecenv.recv()
                env_id = env_id.tolist()

            with self.profile.eval_misc:
                self.global_step += sum(mask)

                o = torch.as_tensor(o)
                o_device = o.to(self.config.device)
                r = torch.as_tensor(r)
                d = torch.as_tensor(d)

            with self.profile.eval_forward, torch.no_grad():
                # TODO: In place-update should be faster. Leaking 7% speed max
                # Also should be using a cuda tensor to index
                if lstm_h is not None:
                    h = lstm_h[:, env_id]
                    c = lstm_c[:, env_id]
                    actions, logprob, _, value, (h, c) = policy(o_device, (h, c))
                    lstm_h[:, env_id] = h
                    lstm_c[:, env_id] = c
                else:
                    actions, logprob, _, value = policy(o_device)

                if self.config.device == "cuda":
                    torch.cuda.synchronize()

            with self.profile.eval_misc:
                value = value.flatten()
                actions = actions.cpu().numpy()
                mask = torch.as_tensor(mask)  # * policy.mask)
                o = o if self.config.cpu_offload else o_device
                if self.config.num_envs == 1:
                    actions = np.expand_dims(actions, 0)
                    logprob = logprob.unsqueeze(0)
                self.experience.store(o, value, actions, logprob, r, d, env_id, mask)

                for i in info:
                    for k, v in pufferlib.utils.unroll_nested_dict(i):
                        if "state/" in k:
                            _, key = k.split("/")
                            self.states[ast.literal_eval(key)].append(v)
                        elif "required_events_count" == k:
                            for count, eid in zip(
                                self.infos["required_events_count"], self.infos["env_id"]
                            ):
                                self.event_tracker[eid] = count
                            self.infos[k].append(v)
                        else:
                            self.infos[k].append(v)

            with self.profile.env:
                self.vecenv.send(actions)

        with self.profile.eval_misc:
            # now for a tricky bit:
            # if we have swarm_frequency, we will migrate the bottom
            # % of envs in the batch (by required events count)
            # and migrate them to a new state at random.
            # Now this has a lot of gotchas and is really unstable
            # E.g. Some envs could just constantly be on the bottom since they're never
            # progressing
            # env id in async queues is the index within self.infos - self.config.num_envs + 1
            if (
                self.config.async_wrapper
                and hasattr(self.config, "swarm_frequency")
                and hasattr(self.config, "swarm_keep_pct")
                and self.epoch % self.config.swarm_frequency == 0
                and "required_events_count" in self.infos
                and self.states
            ):
                # collect the top swarm_keep_pct % of the envs in the batch
                largest = [
                    x[1][0]
                    for x in heapq.nlargest(
                        math.ceil(len(self.event_tracker) * self.config.swarm_keep_pct),
                        enumerate(self.event_tracker.items()),
                        key=lambda x: x[1][0],
                    )
                ]
                waiting_for = []

                # find the envs not in the largest
                to_migrate_keys = set(self.event_tracker.keys()) - set(largest)
                print(f"Migrating {len(to_migrate_keys)} states:")
                # Need a way not to reset the env id counter for the driver env
                # Until then env ids are 1-indexed
                for key in to_migrate_keys:
                    # we store states in a weird format
                    # pull a list of states corresponding to a required event completion state
                    new_state_key = random.choice(list(self.states.keys()))
                    # pull a state within that list
                    new_state = random.choice(self.states[new_state_key])

                    print(f"Environment ID: {key}")
                    print(f"\tEvents count: {self.event_tracker[key]} -> {len(new_state_key)}")
                    print(f"\tNew events: {new_state_key}")
                    self.env_recv_queues[key].put(new_state)
                    waiting_for.append(key)
                    # Now copy the hidden state over
                    # This may be a little slow, but so is this whole process
                    # self.next_lstm_state[0][:, i, :] = self.next_lstm_state[0][:, new_state, :]
                    # self.next_lstm_state[1][:, i, :] = self.next_lstm_state[1][:, new_state, :]
                for i in waiting_for:
                    self.env_send_queues[i].get()
                print("State migration complete")

            self.stats = {}

            for k, v in self.infos.items():
                # Moves into models... maybe. Definitely moves.
                # You could also just return infos and have it in demo
                if "pokemon_exploration_map" in k and self.config.save_overlay is True:
                    if self.epoch % self.config.overlay_interval == 0:
                        overlay = make_pokemon_red_overlay(np.stack(self.infos[k], axis=0))
                        if self.wandb_client is not None:
                            self.stats["Media/aggregate_exploration_map"] = wandb.Image(overlay)
                elif "state" in k:
                    continue

                try:  # TODO: Better checks on log data types
                    self.stats[k] = np.mean(v)
                except:  # noqa: E722
                    continue

            if self.config.verbose:
                self.msg = f"Model Size: {abbreviate(count_params(self.policy))} parameters"
                print_dashboard(
                    self.config.env,
                    self.utilization,
                    self.global_step,
                    self.epoch,
                    self.profile,
                    self.losses,
                    self.stats,
                    self.msg,
                )

        return self.stats, self.infos

    @pufferlib.utils.profile
    def train(self):
        self.losses = Losses()
        losses = self.losses

        with self.profile.train_misc:
            idxs = self.experience.sort_training_data()
            dones_np = self.experience.dones_np[idxs]
            values_np = self.experience.values_np[idxs]
            rewards_np = self.experience.rewards_np[idxs]
            # TODO: bootstrap between segment bounds
            advantages_np = compute_gae(
                dones_np, values_np, rewards_np, self.config.gamma, self.config.gae_lambda
            )
            self.experience.flatten_batch(advantages_np)

        for _ in range(self.config.update_epochs):
            lstm_state = None
            for mb in range(self.experience.num_minibatches):
                with self.profile.train_misc:
                    obs = self.experience.b_obs[mb]
                    obs = obs.to(self.config.device)
                    atn = self.experience.b_actions[mb]
                    log_probs = self.experience.b_logprobs[mb]
                    val = self.experience.b_values[mb]
                    adv = self.experience.b_advantages[mb]
                    ret = self.experience.b_returns[mb]

                with self.profile.train_forward:
                    if self.experience.lstm_h is not None:
                        _, newlogprob, entropy, newvalue, lstm_state = self.policy(
                            obs, state=lstm_state, action=atn
                        )
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                    else:
                        _, newlogprob, entropy, newvalue = self.policy(
                            obs.reshape(-1, *self.vecenv.single_observation_space.shape),
                            action=atn,
                        )

                    if self.config.device == "cuda":
                        torch.cuda.synchronize()

                with self.profile.train_misc:
                    logratio = newlogprob - log_probs.reshape(-1)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > self.config.clip_coef).float().mean()

                    adv = adv.reshape(-1)
                    if self.config.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = val + torch.clamp(
                            newvalue - val,
                            -self.config.vf_clip_coef,
                            self.config.vf_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                    )

                with self.profile.learn:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    if self.config.device == "cuda":
                        torch.cuda.synchronize()

                with self.profile.train_misc:
                    losses.policy_loss += pg_loss.item() / self.experience.num_minibatches
                    losses.value_loss += v_loss.item() / self.experience.num_minibatches
                    losses.entropy += entropy_loss.item() / self.experience.num_minibatches
                    losses.old_approx_kl += old_approx_kl.item() / self.experience.num_minibatches
                    losses.approx_kl += approx_kl.item() / self.experience.num_minibatches
                    losses.clipfrac += clipfrac.item() / self.experience.num_minibatches

            if self.config.target_kl is not None:
                if approx_kl > self.config.target_kl:
                    break

        with self.profile.train_misc:
            if self.config.anneal_lr:
                frac = 1.0 - self.global_step / self.config.total_timesteps
                lrnow = frac * self.config.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            y_pred = self.experience.values_np
            y_true = self.experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            losses.explained_variance = explained_var
            self.epoch += 1

            done_training = self.global_step >= self.config.total_timesteps
            if self.profile.update(self) or done_training:
                if self.config.verbose:
                    print_dashboard(
                        self.config.env,
                        self.utilization,
                        self.global_step,
                        self.epoch,
                        self.profile,
                        self.losses,
                        self.stats,
                        self.msg,
                    )

                if (
                    self.wandb_client is not None
                    and self.global_step > 0
                    and time.time() - self.last_log_time > 5.0
                ):
                    self.last_log_time = time.time()
                    self.wandb_client.log(
                        {
                            "Overview/SPS": self.profile.SPS,
                            "Overview/agent_steps": self.global_step,
                            "Overview/learning_rate": self.optimizer.param_groups[0]["lr"],
                            **{f"environment/{k}": v for k, v in self.stats.items()},
                            **{f"losses/{k}": v for k, v in self.losses.__dict__.items()},
                            **{f"performance/{k}": v for k, v in self.profile},
                        }
                    )

            if self.epoch % self.config.checkpoint_interval == 0 or done_training:
                self.save_checkpoint()
                self.msg = f"Checkpoint saved at update {self.epoch}"

    def close(self):
        self.vecenv.close()
        if self.config.verbose:
            self.utilization.stop()

        if self.wandb_client is not None:
            artifact_name = f"{self.exp_name}_model"
            artifact = wandb.Artifact(artifact_name, type="model")
            model_path = self.save_checkpoint()
            artifact.add_file(model_path)
            self.wandb_client.log_artifact(artifact)
            self.wandb_client.finish()

    def save_checkpoint(self):
        config = self.config
        path = os.path.join(config.data_dir, config.exp_id)
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f"model_{self.epoch:06d}.pt"
        model_path = os.path.join(path, model_name)
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy, model_path)

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "update": self.epoch,
            "model_name": model_name,
            "exp_id": config.exp_id,
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)
        return model_path

    def calculate_loss(self, pg_loss, entropy_loss, v_loss):
        loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

    def done_training(self):
        return self.global_step >= self.config.total_timesteps

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print("Done training.")
        self.save_checkpoint()
        self.close()
        print("Run complete")


class Experience:
    """Flat tensor storage and array views for faster indexing"""

    def __init__(
        self,
        batch_size: int,
        agents_per_batch: int,
        bptt_horizon: int,
        minibatch_size: int,
        obs_shape: tuple[int],
        obs_dtype: np.dtype,
        atn_shape: tuple[int],
        cpu_offload: bool = False,
        device: str = "cuda",
        lstm: torch.nn.LSTM | None = None,
        lstm_total_agents: int = 0,
    ):
        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_dtype]
        pin = device == "cuda" and cpu_offload
        # obs_device = device if not pin else "cpu"
        self.obs = torch.zeros(
            batch_size,
            *obs_shape,
            dtype=obs_dtype,
            pin_memory=pin,
            device=device if not pin else "cpu",
        )
        self.actions = torch.zeros(batch_size, *atn_shape, dtype=int, pin_memory=pin)
        self.logprobs = torch.zeros(batch_size, pin_memory=pin)
        self.rewards = torch.zeros(batch_size, pin_memory=pin)
        self.dones = torch.zeros(batch_size, pin_memory=pin)
        self.truncateds = torch.zeros(batch_size, pin_memory=pin)
        self.values = torch.zeros(batch_size, pin_memory=pin)

        # self.obs_np = np.asarray(self.obs)
        self.actions_np = np.asarray(self.actions)
        self.logprobs_np = np.asarray(self.logprobs)
        self.rewards_np = np.asarray(self.rewards)
        self.dones_np = np.asarray(self.dones)
        self.truncateds_np = np.asarray(self.truncateds)
        self.values_np = np.asarray(self.values)

        self.lstm_h = self.lstm_c = None
        if lstm is not None:
            assert lstm_total_agents > 0
            shape = (lstm.num_layers, lstm_total_agents, lstm.hidden_size)
            self.lstm_h = torch.zeros(shape).to(device)
            self.lstm_c = torch.zeros(shape).to(device)

        num_minibatches = batch_size / minibatch_size
        self.num_minibatches = int(num_minibatches)
        if self.num_minibatches != num_minibatches:
            raise ValueError("batch_size must be divisible by minibatch_size")

        minibatch_rows = minibatch_size / bptt_horizon
        self.minibatch_rows = int(minibatch_rows)
        if self.minibatch_rows != minibatch_rows:
            raise ValueError("minibatch_size must be divisible by bptt_horizon")

        self.batch_size = batch_size
        self.bptt_horizon = bptt_horizon
        self.minibatch_size = minibatch_size
        self.device = device
        self.sort_keys = []
        self.ptr = 0
        self.step = 0

    @property
    def full(self):
        return self.ptr >= self.batch_size

    def store(
        self,
        obs: torch.Tensor,
        value: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        env_id: torch.Tensor,
        mask: torch.Tensor,
    ):
        # Mask learner and Ensure indices do not exceed batch size
        ptr = self.ptr
        indices = torch.where(mask)[0].numpy()[: self.batch_size - ptr]
        end = ptr + len(indices)

        self.obs[ptr:end] = obs.to(self.obs.device)[indices]
        self.values_np[ptr:end] = value.cpu().numpy()[indices]
        self.actions_np[ptr:end] = action[indices]
        self.logprobs_np[ptr:end] = logprob.cpu().numpy()[indices]
        self.rewards_np[ptr:end] = reward.cpu().numpy()[indices]
        self.dones_np[ptr:end] = done.cpu().numpy()[indices]
        self.sort_keys.extend([(env_id[i], self.step) for i in indices])
        self.ptr = end
        self.step += 1

    def sort_training_data(self):
        idxs = np.asarray(sorted(range(len(self.sort_keys)), key=self.sort_keys.__getitem__))
        self.b_idxs_obs = (
            torch.as_tensor(
                idxs.reshape(
                    self.minibatch_rows, self.num_minibatches, self.bptt_horizon
                ).transpose(1, 0, -1)
            )
            .to(self.obs.device)
            .long()
        )
        self.b_idxs = self.b_idxs_obs.to(self.device)
        self.b_idxs_flat = self.b_idxs.reshape(self.num_minibatches, self.minibatch_size)
        self.sort_keys = []
        self.ptr = 0
        self.step = 0
        return idxs

    def flatten_batch(self, advantages_np: np.ndarray):
        advantages = torch.from_numpy(advantages_np).to(self.device)
        b_idxs, b_flat = self.b_idxs, self.b_idxs_flat
        self.b_actions = self.actions.to(self.device, non_blocking=True)
        self.b_logprobs = self.logprobs.to(self.device, non_blocking=True)
        self.b_dones = self.dones.to(self.device, non_blocking=True)
        self.b_values = self.values.to(self.device, non_blocking=True)
        self.b_advantages = (
            advantages.reshape(self.minibatch_rows, self.num_minibatches, self.bptt_horizon)
            .transpose(0, 1)
            .reshape(self.num_minibatches, self.minibatch_size)
        )
        self.returns_np = advantages_np + self.values_np
        self.b_obs = self.obs[self.b_idxs_obs]
        self.b_actions = self.b_actions[b_idxs].contiguous()
        self.b_logprobs = self.b_logprobs[b_idxs]
        self.b_dones = self.b_dones[b_idxs]
        self.b_values = self.b_values[b_flat]
        self.b_returns = self.b_advantages + self.b_values


ROUND_OPEN = rich.box.Box(
    "╭──╮\n"  # noqa: F401
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "│  │\n"
    "╰──╯\n"
)

c1 = "[bright_cyan]"
c2 = "[white]"
c3 = "[cyan]"
b1 = "[bright_cyan]"
b2 = "[bright_white]"


def abbreviate(num):
    if num < 1e3:
        return f"{b2}{num:.0f}"
    elif num < 1e6:
        return f"{b2}{num/1e3:.1f}{c2}k"
    elif num < 1e9:
        return f"{b2}{num/1e6:.1f}{c2}m"
    elif num < 1e12:
        return f"{b2}{num/1e9:.1f}{c2}b"
    else:
        return f"{b2}{num/1e12:.1f}{c2}t"


def duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return (
        f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s"
        if h
        else f"{b2}{m}{c2}m {b2}{s}{c2}s"
        if m
        else f"{b2}{s}{c2}s"
    )


def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100 * time / uptime - 1e-5)
    return f"{c1}{name}", duration(time), f"{b2}{percent:2d}%"


# TODO: Add env name to print_dashboard
def print_dashboard(
    env_name: str,
    utilization: Utilization,
    global_step: int,
    epoch: int,
    profile: Profile,
    losses: Losses,
    stats,
    msg: str,
    clear: bool = False,
    max_stats=None,
):
    if not max_stats:
        max_stats = [0]
    console = Console()
    if clear:
        console.clear()

    dashboard = Table(box=ROUND_OPEN, expand=True, show_header=False, border_style="bright_cyan")

    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)
    cpu_percent = np.mean(utilization.cpu_util)
    dram_percent = np.mean(utilization.cpu_mem)
    gpu_percent = np.mean(utilization.gpu_util)
    vram_percent = np.mean(utilization.gpu_mem)
    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=13)
    table.add_column(justify="right", width=13)
    table.add_row(
        f":blowfish: {c1}PufferLib {b2}1.0.0",
        f"{c1}CPU: {c3}{cpu_percent:.1f}%",
        f"{c1}GPU: {c3}{gpu_percent:.1f}%",
        f"{c1}DRAM: {c3}{dram_percent:.1f}%",
        f"{c1}VRAM: {c3}{vram_percent:.1f}%",
    )

    s = Table(box=None, expand=True)
    s.add_column(f"{c1}Summary", justify="left", vertical="top", width=16)
    s.add_column(f"{c1}Value", justify="right", vertical="top", width=8)
    s.add_row(f"{c2}Environment", f"{b2}{env_name}")
    s.add_row(f"{c2}Agent Steps", abbreviate(global_step))
    s.add_row(f"{c2}SPS", abbreviate(profile.SPS))
    s.add_row(f"{c2}Epoch", abbreviate(epoch))
    s.add_row(f"{c2}Uptime", duration(profile.uptime))
    s.add_row(f"{c2}Remaining", duration(profile.remaining))

    p = Table(box=None, expand=True, show_header=False)
    p.add_column(f"{c1}Performance", justify="left", width=10)
    p.add_column(f"{c1}Time", justify="right", width=8)
    p.add_column(f"{c1}%", justify="right", width=4)
    p.add_row(*fmt_perf("Evaluate", profile.eval_time, profile.uptime))
    p.add_row(*fmt_perf("  Forward", profile.eval_forward_time, profile.uptime))
    p.add_row(*fmt_perf("  Env", profile.env_time, profile.uptime))
    p.add_row(*fmt_perf("  Misc", profile.eval_misc_time, profile.uptime))
    p.add_row(*fmt_perf("Train", profile.train_time, profile.uptime))
    p.add_row(*fmt_perf("  Forward", profile.train_forward_time, profile.uptime))
    p.add_row(*fmt_perf("  Learn", profile.learn_time, profile.uptime))
    p.add_row(*fmt_perf("  Misc", profile.train_misc_time, profile.uptime))

    l = Table(  # noqa: E741
        box=None,
        expand=True,
    )
    l.add_column(f"{c1}Losses", justify="left", width=16)
    l.add_column(f"{c1}Value", justify="right", width=8)
    for metric, value in losses.__dict__.items():
        l.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(s, p, l)
    dashboard.add_row(monitor)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    left = Table(box=None, expand=True)
    right = Table(box=None, expand=True)
    table.add_row(left, right)
    left.add_column(f"{c1}User Stats", justify="left", width=20)
    left.add_column(f"{c1}Value", justify="right", width=10)
    right.add_column(f"{c1}User Stats", justify="left", width=20)
    right.add_column(f"{c1}Value", justify="right", width=10)
    i = 0
    for metric, value in stats.items():
        try:  # Discard non-numeric values
            int(value)
        except:  # noqa: E722
            continue

        u = left if i % 2 == 0 else right
        u.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")
        i += 1

    for i in range(max_stats[0] - i):
        u = left if i % 2 == 0 else right
        u.add_row("", "")

    max_stats[0] = max(max_stats[0], i)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    table.add_row(f" {c1}Message: {c2}{msg}")

    with console.capture() as capture:
        console.print(dashboard)

    print("\033[0;0H" + capture.get())
