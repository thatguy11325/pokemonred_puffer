import json
import math
import multiprocessing as mp
import os
import pprint
import re
from typing import Annotated

import carbs.utils
import sweeps
import typer
from carbs import (
    CARBS,
    CARBSParams,
    LogitSpace,
    ObservationInParam,
    Param,
    ParamDictType,
    ParamType,
    WandbLoggingParams,
)
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from sweeps import RunState

import wandb
from pokemonred_puffer import train
from pokemonred_puffer.environment import RedGymEnv

app = typer.Typer(pretty_exceptions_enable=False)


def sweep_config_to_params(
    base_config: DictConfig | int | float | bool | None, sweep_config: DictConfig, prefix: str = ""
) -> list[Param]:
    res = []
    for k, v in sweep_config.items():
        # A little hacky. Maybe I should not make this all config based
        if k.startswith("carbs.utils"):
            param_class = getattr(carbs.utils, k.split(".")[-1])
            res += [
                Param(
                    prefix.removesuffix("-").removeprefix("-"),
                    param_class(**v),
                    base_config
                    if base_config is not None
                    else (
                        (v["max"] + v["min"]) // 2
                        if param_class == "LinearSpace"
                        else math.sqrt(v["max"] * v["min"])
                    ),
                )
            ]
        elif isinstance(v, DictConfig):
            res += sweep_config_to_params(base_config.get(k, None), v, prefix=prefix + "-" + k)
        else:
            print(type(v))
    return res


def update_base_config_by_key(base_config: DictConfig, key: str, value: ParamType) -> DictConfig:
    new_config = base_config.copy()
    keys = key.split("-", 1)
    if len(keys) == 1:
        new_config[keys[0]] = value
    else:
        new_config[keys[0]] = update_base_config_by_key(new_config[keys[0]], keys[1], value)
    return new_config


def update_base_config(base_config: DictConfig, suggestion: ParamDictType) -> DictConfig:
    new_config = base_config.copy()
    for k, v in suggestion.items():
        new_config = update_base_config_by_key(new_config, k, v)
    return new_config


@app.command()
def launch_sweep(
    base_config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = "config.yaml",
    sweep_config: Annotated[
        DictConfig,
        typer.Option(
            help="CARBS sweep config. settings must match base config.", parser=OmegaConf.load
        ),
    ] = "sweep-config.yaml",
    sweep_name: Annotated[str, typer.Option(help="Sweep name")] = "PokeSweep",
    sweep_id: Annotated[
        str | None,
        typer.Option(
            help="Sweep id to use. If specified, a previous sweep will be resumed. "
            "N.B. The sweep and base config MUST BE THE SAME"
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Attempts to start CARBS, but not wandb")] = False,
):
    console = Console()
    params = sweep_config_to_params(base_config, sweep_config)
    params_keys = {p.name for p in params}
    for param in params:
        print(f"Checking param: {param}")
        if isinstance(param.space, LogitSpace):
            assert (
                0.0
                <= param.space.min
                < param.search_center - param.space.scale
                < param.search_center + param.space.scale
                < param.space.max
                <= 1.0
            ), (
                "0.0 "
                f"<= {param.space.min} "
                f"< {param.search_center} - {param.space.scale} "
                f"< {param.search_center} + {param.space.scale} "
                f"< {param.space.max} "
                f"<= 1.0"
            )
        else:
            assert (
                param.space.min
                < param.search_center - param.space.scale
                < param.search_center + param.space.scale
                < param.space.max
            ), (
                f"{param.space.min} "
                f"< {param.search_center} - {param.space.scale} "
                f"< {param.search_center} + {param.space.scale} "
                f"< {param.space.max}"
            )

    if sweep_id:
        config = CARBSParams(
            better_direction_sign=1,
            is_wandb_logging_enabled=True,
            wandb_params=WandbLoggingParams(
                project_name="Pokemon",
                run_id=sweep_id,
                run_name=sweep_name,
                root_dir=f"carbs/checkpoints/{sweep_id}",
            ),
            resample_frequency=5,
            num_random_samples=len(params),
            checkpoint_dir=f"carbs/checkpoints/{sweep_id}",
        )

        carbs = CARBS(config=config, params=params)
        # for wandb
        # runname = entity/project/run_id
        # find most recent file in checkpoint dir
        experiment_dir = f"carbs/checkpoints/{sweep_id}/{sweep_name}"
        saves = [
            save_filename
            for save_filename in os.listdir(experiment_dir)
            if re.match(r"carbs_\d+obs.pt", save_filename)
        ]

        if saves:
            # sort by the middle int and take the highest value
            # dont need split, could also use a regex group
            print(f"Found saves {saves}")
            save_filename = sorted(
                saves,
                key=lambda x: int(x.replace("carbs_", "").replace("obs.pt", "")),
                reverse=True,
            )[0]
            print(f"Warm starting carbs from save {save_filename}")
            carbs.warm_start(
                filename=os.path.join(experiment_dir, save_filename),
                is_prior_observation_valid=True,
            )
        else:
            print("Found no saves. Carbs will not be warm started.")

    if not sweep_id and not dry_run:
        sweep_id = wandb.sweep(
            sweep={
                "name": sweep_name,
                "controller": {"type": "local"},
                "parameters": {p.name: {"min": p.space.min, "max": p.space.max} for p in params},
                "metric": {
                    "name": "environment/stats/required_count",
                    "goal": "maximize",
                },
                "command": ["${args_json}"],
            },
            entity=base_config.wandb.entity,
            project=base_config.wandb.project,
        )
        config = CARBSParams(
            better_direction_sign=1,
            is_wandb_logging_enabled=True,
            wandb_params=WandbLoggingParams(
                project_name="Pokemon",
                run_id=sweep_id,
                run_name=sweep_name,
                root_dir=f"carbs/checkpoints/{sweep_id}",
            ),
            resample_frequency=5,
            num_random_samples=len(params),
            checkpoint_dir=f"carbs/checkpoints/{sweep_id}",
        )
        carbs = CARBS(config=config, params=params)
        os.makedirs(os.path.join(config.checkpoint_dir, carbs.experiment_name), exist_ok=True)

        carbs._autosave()

    pprint.pprint(params)
    if dry_run:
        carbs.suggest()
        return

    sweep = wandb.controller(
        sweep_id_or_config=sweep_id,
        entity=base_config.wandb.entity,
        project=base_config.wandb.project,
    )

    console.print(f"Beginning sweep with id {sweep_id}", style="bold")
    console.print(
        f"On all nodes please run python -m pokemonred_puffer.sweep launch-agent {sweep_id}",
        style="bold",
    )

    finished = []
    while not sweep.done():
        # Taken from sweep.schedule. Limits runs to only one at a time.
        # Only one run will be scheduled at a time
        sweep._step()
        if not (sweep._controller and sweep._controller.get("schedule")):
            suggestion = carbs.suggest()
            run = sweeps.SweepRun(
                config={k: {"value": v} for k, v in suggestion.suggestion.items()}
            )
            sweep.schedule(run)
        # without this nothing updates...
        sweep_obj = sweep._sweep_obj
        if runs := sweep_obj["runs"]:
            for run in runs:
                if run["state"] == RunState.running.value:
                    pass
                elif (
                    run["state"]
                    in [
                        RunState.failed.value,
                        RunState.finished.value,
                        RunState.crashed.value,
                    ]
                    and run["name"] not in finished
                ):
                    finished.append(run["name"])
                    if summaryMetrics_json := run.get("summaryMetrics", None):
                        summary_metrics = json.loads(summaryMetrics_json)
                        if (
                            "environment/stats/required_count" in summary_metrics
                            and "performance/uptime" in summary_metrics
                            # Only count agents that have run more than 1M steps
                            and "Overview/agent_steps" in summary_metrics
                            and summary_metrics["Overview/agent_steps"] > 1e6
                        ):
                            obs_in = ObservationInParam(
                                input={
                                    k: v["value"]
                                    for k, v in json.loads(run["config"]).items()
                                    if k in params_keys
                                },
                                # TODO: try out other stats like required count
                                output=summary_metrics["environment/stats/required_count"],
                                cost=summary_metrics["performance/uptime"],
                            )
                            carbs.observe(obs_in)
                            # Because wandb stages the artifacts we have to keep these files
                            # dangling around wasting good disk space.
                            # carbs.save_to_file(hash(tuple(finished)) + ".pt", upload_to_wandb=True)
                elif run["state"] == RunState.pending:
                    print(f"PENDING RUN FOUND {run['name']}")
        sweep.print_status()


@app.command()
def launch_agent(
    sweep_id: str,
    base_config: Annotated[
        DictConfig,
        typer.Option(help="Base configuration. MUST MATCH PRIMARY NODE.", parser=OmegaConf.load),
    ] = "config.yaml",
    debug: bool = False,
):
    def _fn():
        agent_config: DictConfig = OmegaConf.create(
            {
                k: v.value
                for k, v in OmegaConf.load(os.environ["WANDB_SWEEP_PARAM_PATH"]).items()
                if k != "wandb_version"
            }
        )
        agent_config = update_base_config(base_config, agent_config)
        try:
            train.train(config=agent_config, debug=debug, track=True)
        except Exception as e:
            print(f"Exception in training: {e!r}")

    for _ in range(99999):
        # Manually reset the env id counter between runs
        RedGymEnv.env_id.buf[0] = 0
        RedGymEnv.env_id.buf[1] = 0
        RedGymEnv.env_id.buf[2] = 0
        RedGymEnv.env_id.buf[3] = 0
        proc = mp.Process(
            target=wandb.agent,
            kwargs=dict(
                sweep_id=sweep_id,
                entity=base_config.wandb.entity,
                project=base_config.wandb.project,
                function=_fn,
                count=1,
            ),
        )
        proc.start()
        proc.join()


if __name__ == "__main__":
    app()
