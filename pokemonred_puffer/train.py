import argparse
import functools
import importlib
import os
import sys
from multiprocessing import Queue
from types import ModuleType
from typing import Any, Callable
import uuid

import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.postprocess
import pufferlib.utils
import pufferlib.vector
import yaml

import wandb
from pokemonred_puffer import cleanrl_puffer
from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL
from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.wrappers.async_io import AsyncWrapper


def make_policy(env, policy_name, args):
    policy_module_name, policy_class_name = policy_name.split(".")
    policy_module = importlib.import_module(f"pokemonred_puffer.policies.{policy_module_name}")
    policy_class = getattr(policy_module, policy_class_name)

    policy = policy_class(env, **args.policies[policy_name]["policy"])
    if args.train.use_rnn:
        rnn_config = args.policies[policy_name]["rnn"]
        policy_class = getattr(policy_module, rnn_config["name"])
        policy = policy_class(env, policy, **rnn_config["args"])
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args.train.device)


# TODO: Replace with Pydantic or Spock parser
def load_from_config(args: argparse.ArgumentParser):
    with open(args["yaml"]) as f:
        config = yaml.safe_load(f)

    default_keys = ["env", "train", "policies", "rewards", "wrappers", "wandb"]
    defaults = {key: config.get(key, {}) for key in default_keys}

    # Package and subpackage (environment) configs
    debug_config = config.get("debug", {}) if args["debug"] else {}
    # This is overly complicated. Clean it up. Or remove configs entirely
    # if we're gonna start creating an ersatz programming language.
    wrappers_config = {}
    for wrapper in config["wrappers"][args["wrappers_name"]]:
        for k, v in wrapper.items():
            wrappers_config[k] = v
    reward_config = config["rewards"][args["reward_name"]]
    policy_config = config["policies"][args["policy_name"]]

    combined_config = {}
    for key in default_keys:
        policy_subconfig = policy_config.get(key, {})
        reward_subconfig = reward_config.get(key, {})
        wrappers_subconfig = wrappers_config.get(key, {})
        debug_subconfig = debug_config.get(key, {})

        # Order of precedence: debug > wrappers > rewards > policy > defaults
        combined_config[key] = (
            defaults[key]
            | policy_subconfig
            | reward_subconfig
            | wrappers_subconfig
            | debug_subconfig
        )
    return pufferlib.namespace(**combined_config)


def make_env_creator(
    wrapper_classes: list[tuple[str, ModuleType]],
    reward_class: RedGymEnv,
    async_wrapper: bool = True,
) -> Callable[[pufferlib.namespace, pufferlib.namespace], pufferlib.emulation.GymnasiumPufferEnv]:
    def env_creator(
        env_config: pufferlib.namespace,
        wrappers_config: list[dict[str, Any]],
        reward_config: pufferlib.namespace,
        async_config: dict[str, Queue] | None = None,
    ) -> pufferlib.emulation.GymnasiumPufferEnv:
        env = reward_class(env_config, reward_config)
        for cfg, (_, wrapper_class) in zip(wrappers_config, wrapper_classes):
            env = wrapper_class(env, pufferlib.namespace(**[x for x in cfg.values()][0]))
        if async_wrapper and async_config:
            env = AsyncWrapper(env, async_config["send_queues"], async_config["recv_queues"])
        # env = pufferlib.postprocess.EpisodeStats(env)
        return pufferlib.emulation.GymnasiumPufferEnv(env=env)

    return env_creator


# Returns env_creator, agent_creator
def setup_agent(
    wrappers: list[str], reward_name: str, async_wrapper: bool = True
) -> Callable[[pufferlib.namespace, pufferlib.namespace], pufferlib.emulation.GymnasiumPufferEnv]:
    # TODO: Make this less dependent on the name of this repo and its file structure
    wrapper_classes = [
        (
            k,
            getattr(
                importlib.import_module(f"pokemonred_puffer.wrappers.{k.split('.')[0]}"),
                k.split(".")[1],
            ),
        )
        for wrapper_dicts in wrappers
        for k in wrapper_dicts.keys()
    ]
    reward_module, reward_class_name = reward_name.split(".")
    reward_class = getattr(
        importlib.import_module(f"pokemonred_puffer.rewards.{reward_module}"), reward_class_name
    )
    # NOTE: This assumes reward_module has RewardWrapper(RedGymEnv) class
    env_creator = make_env_creator(wrapper_classes, reward_class, async_wrapper)

    return env_creator


def update_args(args: argparse.Namespace):
    args = pufferlib.namespace(**args)

    args.track = args.track
    args.env.gb_path = args.rom_path

    if args.vectorization == "serial" or args.debug:
        args.vectorization = pufferlib.vector.Serial
    elif args.vectorization == "multiprocessing":
        args.vectorization = pufferlib.vector.Multiprocessing

    return args


def init_wandb(args, resume=True):
    assert args.wandb.project is not None, "Please set the wandb project in config.yaml"
    assert args.wandb.entity is not None, "Please set the wandb entity in config.yaml"
    wandb_kwargs = {
        "id": args.exp_name or wandb.util.generate_id(),
        "project": args.wandb.project,
        "entity": args.wandb.entity,
        "group": args.wandb.group,
        "config": {
            "cleanrl": args.train,
            "env": args.env,
            "reward_module": args.reward_name,
            "policy_module": args.policy_name,
            "reward": args.rewards[args.reward_name],
            "policy": args.policies[args.policy_name],
            "wrappers": args.wrappers[args.wrappers_name],
            "recurrent": "recurrent" in args.policies[args.policy_name],
        },
        "name": args.exp_name,
        "monitor_gym": True,
        "save_code": True,
        "resume": resume,
    }
    return wandb.init(**wandb_kwargs)


def train(
    args: pufferlib.namespace,
    env_creator: Callable,
    wandb_client: wandb.wandb_sdk.wandb_run.Run | None,
):
    vec = args.vectorization
    if vec == "serial":
        vec = pufferlib.vector.Serial
    elif vec == "multiprocessing":
        vec = pufferlib.vector.Multiprocessing
    elif vec == "ray":
        vec = pufferlib.vector.Ray

    # TODO: Remove the +1 once the driver env doesn't permanently increase the env id
    env_send_queues = [Queue() for _ in range(args.train.num_envs + 1)]
    env_recv_queues = [Queue() for _ in range(args.train.num_envs + 1)]

    vecenv = pufferlib.vector.make(
        env_creator,
        env_kwargs={
            "env_config": args.env,
            "wrappers_config": args.wrappers[args.wrappers_name],
            "reward_config": args.rewards[args.reward_name]["reward"],
            "async_config": {"send_queues": env_send_queues, "recv_queues": env_recv_queues},
        },
        num_envs=args.train.num_envs,
        num_workers=args.train.num_workers,
        batch_size=args.train.env_batch_size,
        zero_copy=args.train.zero_copy,
        backend=vec,
    )
    policy = make_policy(vecenv.driver_env, args.policy_name, args)

    args.train.env = "Pokemon Red"
    with CleanPuffeRL(
        exp_name=args.exp_name,
        config=args.train,
        vecenv=vecenv,
        policy=policy,
        env_recv_queues=env_recv_queues,
        env_send_queues=env_send_queues,
        wandb_client=wandb_client,
    ) as trainer:
        while not trainer.done_training():
            trainer.evaluate()
            trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse environment argument", add_help=False)
    parser.add_argument("--yaml", default="config.yaml", help="Configuration file to use")
    parser.add_argument(
        "-p",
        "--policy-name",
        default="multi_convolutional.MultiConvolutionalPolicy",
        help="Policy module to use in policies.",
    )
    parser.add_argument(
        "-r",
        "--reward-name",
        default="baseline.BaselineRewardEnv",
        help="Reward module to use in rewards",
    )
    parser.add_argument(
        "-w",
        "--wrappers-name",
        type=str,
        default="baseline",
        help="Wrappers to use _in order of instantiation_.",
    )
    # TODO: Add evaluate
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "autotune", "evaluate"]
    )
    parser.add_argument(
        "--eval-model-path", type=str, default=None, help="Path to model to evaluate"
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Resume from experiment")
    parser.add_argument("--rom-path", default="red.gb")
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--vectorization",
        type=str,
        default="multiprocessing",
        choices=["serial", "multiprocessing"],
        help="Vectorization method (serial, multiprocessing)",
    )

    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__
    config = load_from_config(args)

    # Generate argparse menu from config
    # This is also a reason for Spock/Argbind/OmegaConf/pydantic-cli
    for name, sub_config in config.items():
        args[name] = {}
        for key, value in sub_config.items():
            data_key = f"{name}.{key}"
            cli_key = f"--{data_key}".replace("_", "-")
            if isinstance(value, bool) and value is False:
                action = "store_false"
                parser.add_argument(cli_key, default=value, action="store_true")
                clean_parser.add_argument(cli_key, default=value, action="store_true")
            elif isinstance(value, bool) and value is True:
                data_key = f"{name}.no_{key}"
                cli_key = f"--{data_key}".replace("_", "-")
                parser.add_argument(cli_key, default=value, action="store_false")
                clean_parser.add_argument(cli_key, default=value, action="store_false")
            else:
                parser.add_argument(cli_key, default=value, type=type(value))
                clean_parser.add_argument(cli_key, default=value, metavar="", type=type(value))

            args[name][key] = getattr(parser.parse_known_args()[0], data_key)
        args[name] = pufferlib.namespace(**args[name])
    clean_parser.parse_args(sys.argv[1:])
    args = update_args(args)
    args.train.exp_id = f"pokemon-red-{str(uuid.uuid4())[:8]}"

    async_wrapper = args.train.async_wrapper
    env_creator = setup_agent(args.wrappers[args.wrappers_name], args.reward_name, async_wrapper)

    wandb_client = None
    if args.track:
        wandb_client = init_wandb(args)

    if args.mode == "train":
        train(args, env_creator, wandb_client)
    elif args.mode == "autotune":
        env_kwargs = {
            "env_config": args.env,
            "wrappers_config": args.wrappers[args.wrappers_name],
            "reward_config": args.rewards[args.reward_name]["reward"],
            "async_config": {},
        }
        pufferlib.vector.autotune(
            functools.partial(env_creator, **env_kwargs), batch_size=args.train.env_batch_size
        )
    elif args.mode == "evaluate":
        env_kwargs = {
            "env_config": args.env,
            "wrappers_config": args.wrappers[args.wrappers_name],
            "reward_config": args.rewards[args.reward_name]["reward"],
            "async_config": {},
        }
        try:
            cleanrl_puffer.rollout(
                env_creator,
                env_kwargs,
                agent_creator=make_policy,
                agent_kwargs={"args": args},
                model_path=args.eval_model_path,
                device=args.train.device,
            )
        except KeyboardInterrupt:
            os._exit(0)
