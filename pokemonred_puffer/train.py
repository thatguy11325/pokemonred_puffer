import argparse
import importlib
import inspect
import sys
import uuid
from pathlib import Path

import pufferlib
import pufferlib.utils
import torch
import wandb
import yaml

from pokemonred_puffer import eval
from pokemonred_puffer.cleanrl_puffer import CleanPuffeRL, rollout
from pokemonred_puffer.env_creator import env_creator 
from pokemonred_puffer.policies import Recurrent, Policy 


# TODO: Replace with Pydantic or Spock parser
def load_from_config(yaml_path, env):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    assert env in config, f'"{env}" not found in config.yaml. '
    "Uncommon environments that are part of larger packages may not have their own config. "
    "Specify these manually using the parent package, e.g. --config atari --env MontezumasRevengeNoFrameskip-v4."

    default_keys = [
        "env",
        "train",
        "policy",
        "recurrent",
        "sweep_metadata",
        "sweep_metric",
        "sweep",
    ]
    defaults = {key: config.get(key, {}) for key in default_keys}

    # Package and subpackage (environment) configs
    env_config = config[env]
    pkg = env_config["package"]
    pkg_config = config[pkg]

    combined_config = {}
    for key in default_keys:
        env_subconfig = env_config.get(key, {})
        pkg_subconfig = pkg_config.get(key, {})

        # Override first with pkg then with env configs
        combined_config[key] = {**defaults[key], **pkg_subconfig, **env_subconfig}

    return pkg, pufferlib.namespace(**combined_config)


def make_policy(env, env_module, args):
    policy = env_module.Policy(env, **args.policy)
    if args.force_recurrence or env_module.Recurrent is not None:
        policy = env_module.Recurrent(env, policy, **args.recurrent)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    mode = "default"
    if args.train.device == "cuda":
        mode = "reduce-overhead"
    policy = policy.to(args.train.device, non_blocking=True)
    policy.get_value = torch.compile(policy.get_value, mode=mode)
    policy.get_action_and_value = torch.compile(policy.get_action_and_value, mode=mode)
    return policy


def init_wandb(args, env_module):
    # os.environ["WANDB_SILENT"] = "true"

    wandb.init(
        id=args.exp_name or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            "cleanrl": args.train,
            "env": args.env,
            "policy": args.policy,
            "recurrent": args.recurrent,
        },
        name=args.config,
        monitor_gym=True,
        save_code=True,
        resume=True,
    )
    return wandb.run.id


def sweep(args, env_module, make_env):
    import wandb

    sweep_id = wandb.sweep(sweep=args.sweep, project="pufferlib")

    def main():
        try:
            args.exp_name = init_wandb(args, env_module)
            if hasattr(wandb.config, "train"):
                # TODO: Add update method to namespace
                print(args.train.__dict__)
                print(wandb.config.train)
                args.train.__dict__.update(dict(wandb.config.train))
            train(args, env_module, make_env)
        except Exception as e:
            import traceback

            traceback.print_exc()

    wandb.agent(sweep_id, main, count=20)


def get_init_args(fn):
    if fn is None:
        return {}

    sig = inspect.signature(fn)
    args = {}
    for name, param in sig.parameters.items():
        if name in ("self", "env", "policy"):
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            args[name] = param.default if param.default is not inspect.Parameter.empty else None
    return args


def train(args, env_module, make_env):
    with CleanPuffeRL(
        config=args.train,
        agent_creator=make_policy,
        agent_kwargs={"env_module": env_module, "args": args},
        env_creator=make_env,
        env_creator_kwargs=args.env,
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
        "--config", type=str, default="pokemon_red", help="Configuration in config.yaml to use"
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sweep", "evaluate"])
    parser.add_argument(
        "--eval-model-path", type=str, default=None, help="Path to model to evaluate"
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        help="Enable/Disable render during evaluate",
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Resume from experiment")
    parser.add_argument(
        "--vectorization",
        type=str,
        default="serial",
        choices=["serial", "multiprocessing"],
        help="Vectorization method (serial, multiprocessing)",
    )
    parser.add_argument("--wandb-entity", type=str, default="thatguy11325", help="WandB entity")
    parser.add_argument("--wandb-project", type=str, default="pokemonred", help="WandB project")
    parser.add_argument("--wandb-group", type=str, default="pokemonred", help="WandB group")
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    parser.add_argument(
        "--force-recurrence",
        action="store_true",
        help="Force model to be recurrent, regardless of defaults",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rom-path", default="red.gb")
    parser.add_argument("--state-path", default="pyboy_states/Bulbasaur.state")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fast-video", action="store_true")
    parser.add_argument("--frame-stacks", type=int, default=1)
    parser.add_argument(
        "--policy",
        choices=["MultiInputPolicy", "CnnPolicy", "CnnLstmPolicy", "MlpLstmPolicy"],
        default="CnnLstmPolicy",
    )
    parser.add_argument("--sess-id", type=str, default=str(uuid.uuid4())[:8])
    parser.add_argument("--perfect-ivs", action="store_true", default=False, help="Enable perfect IVs")

    clean_parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_known_args()[0].__dict__
    parsed_args = parser.parse_args()
    pkg, config = load_from_config(args["yaml"], args["config"])

    # TODO: Adapt to Puffer's config style
    sess_path = Path(f"session_{parsed_args.sess_id}")
    env_config = {
        "headless": parsed_args.headless,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": parsed_args.state_path,
        "max_steps": 3_000_000,
        "print_rewards": True,
        "save_video": parsed_args.save_video,
        "fast_video": parsed_args.fast_video,
        "session_path": sess_path,
        "gb_path": parsed_args.rom_path,
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "extra_buttons": False,
        "explore_weight": 3,  # 2.5
        "explore_npc_weight": 1,  # 2.5
        "frame_stacks": parsed_args.frame_stacks,
        "policy": parsed_args.policy,
        "step_forgetting_factor": {
            "npc": 0.995,
            "hidden_objs": 0.95,
            "coords": 0.9995,
            "map_ids": 0.995,
            "explore": 0.9995
        },
        "forgetting_frequency": 10,
        "perfect_ivs": parsed_args.perfect_ivs
    }

    env_module = importlib.import_module(f"pokemonred_puffer")

    # Update config with environment defaults
    make_env = env_creator(config)
    config.env = {**get_init_args(make_env), **config.env}
    config.policy = {**get_init_args(env_module.policies.Policy.__init__), **config.policy}
    config.recurrent = {**get_init_args(env_module.policies.Recurrent.__init__), **config.recurrent}

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
    args = pufferlib.namespace(**args)

    if args.vectorization == "serial":
        args.vectorization = pufferlib.vectorization.Serial
    elif args.vectorization == "multiprocessing":
        args.vectorization = pufferlib.vectorization.Multiprocessing

    if args.mode == "sweep":
        args.track = True

    if args.track:
        args.exp_name = init_wandb(args, env_module)

    args.env = env_config
    if args.mode == "train":
        train(args, env_module, make_env)
    elif args.mode == "sweep":
        sweep(args, env_module, make_env)
    elif args.mode == "evaluate" and pkg != "pokemon_red":
        rollout(
            make_env,
            args.env,
            agent_creator=make_policy,
            agent_kwargs={"env_module": env_module, "args": args},
            model_path=args.eval_model_path,
            device=args.train.device,
        )
    elif args.mode == "evaluate" and pkg == "pokemon_red":
        eval.rollout(
            make_env,
            args.env,
            agent_creator=make_policy,
            agent_kwargs={"env_module": env_module, "args": args},
            model_path=args.eval_model_path,
            device=args.train.device,
        )
    elif pkg != "pokemon_red":
        raise ValueError("Mode must be one of train, sweep, or evaluate")
