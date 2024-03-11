import argparse
import importlib
import pathlib
import sys
import time
from types import ModuleType
from typing import Any, Callable

import gymnasium as gym
import torch
import wandb
import yaml

import pufferlib
import pufferlib.utils
from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL, rollout
from pokemonred_puffer.environment import RedGymEnv


# TODO: Replace with Pydantic or Spock parser
def load_from_config(
    yaml_path: str | pathlib.Path,
    wrappers: str,
    policy: str,
    reward: str,
    debug: bool = False,
):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    default_keys = ["env", "train", "policies", "rewards", "wrappers"]
    defaults = {key: config.get(key, {}) for key in default_keys}

    # Package and subpackage (environment) configs
    debug_config = config.get("debug", {}) if debug else {}
    # This is overly complicated. Clean it up. Or remove configs entirely
    # if we're gonna start creating an ersatz programming language.
    wrappers_config = {}
    for wrapper in config["wrappers"][wrappers]:
        for k, v in wrapper.items():
            wrappers_config[k] = v
    reward_config = config["rewards"][reward]
    policy_config = config["policies"][policy]

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
    wrapper_classes: dict[str, ModuleType],
    reward_class: ModuleType,
) -> Callable[[pufferlib.namespace, pufferlib.namespace], pufferlib.emulation.GymnasiumPufferEnv]:
    def env_creator(
        env_config: pufferlib.namespace,
        wrappers_config: list[dict[str, Any]],
        reward_config: pufferlib.namespace,
    ) -> pufferlib.emulation.GymnasiumPufferEnv:
        env = reward_class(RedGymEnv(env_config), reward_config)
        flattened_wrappers_config = {k: v for d in wrappers_config for k, v in d.items()}
        for wrapper_name, wrapper_class in wrapper_classes.items():
            env = wrapper_class(env, pufferlib.namespace(**flattened_wrappers_config[wrapper_name]))
        return pufferlib.emulation.GymnasiumPufferEnv(
            env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
        )

    return env_creator


# Returns env_creator, agent_creator
def setup_agent(
    wrappers: list[str],
    reward_name: str,
    policy_name: str,
) -> Callable[[pufferlib.namespace, pufferlib.namespace], pufferlib.emulation.GymnasiumPufferEnv]:
    # TODO: Make this less dependent on the name of this repo and its file structure
    wrapper_classes = {
        k: getattr(
            importlib.import_module(f"pokemonred_puffer.wrappers.{k.split('.')[0]}"),
            k.split(".")[1],
        )
        for wrapper_dicts in wrappers
        for k in wrapper_dicts.keys()
    }
    reward_module, reward_class_name = reward_name.split(".")
    reward_class = getattr(
        importlib.import_module(f"pokemonred_puffer.rewards.{reward_module}"), reward_class_name
    )
    # NOTE: This assumes reward_module has RewardWrapper(RedGymEnv) class
    env_creator = make_env_creator(wrapper_classes, reward_class)

    policy_module_name, policy_class_name = policy_name.split(".")
    policy_module = importlib.import_module(f"pokemonred_puffer.policies.{policy_module_name}")
    policy_class = getattr(policy_module, policy_class_name)

    def agent_creator(env: gym.Env, args: pufferlib.namespace):
        policy = policy_class(env, **args.policies[policy_name]["policy"])
        if "recurrent" in args.policies[policy_name]:
            recurrent_args = args.policies[policy_name]["recurrent"]
            recurrent_class_name = recurrent_args["name"]
            del recurrent_args["name"]
            policy = getattr(policy_module, recurrent_class_name)(env, policy, **recurrent_args)
            policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
        else:
            policy = pufferlib.frameworks.cleanrl.Policy(policy)

        if args.train.device == "cuda" and args.train.compile:
            torch.set_float32_matmul_precision(args.train.float32_matmul_precision)
            policy = policy.to(args.train.device, non_blocking=True)
            policy.get_value = torch.compile(policy.get_value, mode=args.train.compile_mode)
            policy.get_action_and_value = torch.compile(
                policy.get_action_and_value, mode=args.train.compile_mode
            )

        return policy

    return env_creator, agent_creator


def update_args(args: argparse.Namespace):
    args = pufferlib.namespace(**args)

    args.track = args.track
    args.env.gb_path = args.rom_path

    if args.vectorization == "serial" or args.debug:
        args.vectorization = pufferlib.vectorization.Serial
    elif args.vectorization == "multiprocessing":
        args.vectorization = pufferlib.vectorization.Multiprocessing

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
            "reward": args.reward,
            "policy": args.policy,
            "wrappers": args.wrappers,
            "recurrent": args.recurrent,
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
    agent_creator: Callable[[gym.Env, pufferlib.namespace], pufferlib.models.Policy],
):
    with CleanPuffeRL(
        config=args.train,
        agent_creator=agent_creator,
        agent_kwargs={"args": args},
        env_creator=env_creator,
        env_creator_kwargs={
            "env_config": args.env,
            "wrappers_config": args.wrappers[args.wrappers_name],
            "reward_config": args.rewards[args.reward_name]["reward"],
        },
        vectorization=args.vectorization,
        exp_name=args.exp_name,
        track=args.track,
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
        help="Policy module to use in policies",
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
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
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
    config = load_from_config(
        args["yaml"], args["wrappers_name"], args["policy_name"], args["reward_name"], args["debug"]
    )

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

    env_creator, agent_creator = setup_agent(
        args.wrappers[args.wrappers_name], args.reward_name, args.policy_name
    )

    if args.track:
        args.exp_name = init_wandb(args).id
    else:
        args.exp_name = f"poke_{time.strftime('%Y%m%d_%H%M%S')}"
    args.env.session_path = args.exp_name

    if args.mode == "train":
        train(args, env_creator, agent_creator)

    elif args.mode == "evaluate":
        # TODO: check if this works
        rollout(
            env_creator=env_creator,
            env_creator_kwargs={"env_config": args.env, "wrappers_config": args.wrappers},
            agent_creator=agent_creator,
            agent_kwargs={"args": args},
            model_path=args.eval_model_path,
            device=args.train.device,
        )
    else:
        raise ValueError("Mode must be one of train or evaluate")
