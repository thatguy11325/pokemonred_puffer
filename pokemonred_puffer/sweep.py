import json
import math
import multiprocessing as mp
import os
from typing import Annotated

import carbs.utils
import sweeps
from sweeps import RunState
import typer
from carbs import (
    CARBS,
    Param,
    ParamDictType,
    ParamType,
    CARBSParams,
    WandbLoggingParams,
    ObservationInParam,
)
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
import wandb

from pokemonred_puffer import train

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
):
    console = Console()
    if not sweep_id:
        config = CARBSParams(
            better_direction_sign=1,
            is_wandb_logging_enabled=False,
            wandb_params=WandbLoggingParams(project_name="Pokemon", run_name=sweep_id),
        )
        params = sweep_config_to_params(base_config, sweep_config)

        carbs = CARBS(config=config, params=params)
        sweep_id = wandb.sweep(
            sweep={
                "name": sweep_name,
                "controller": {"type": "local"},
                "parameters": {p.name: {"min": p.space.min, "max": p.space.max} for p in params},
                "metric": {
                    "name": "environment/stats/required_count",
                    "goal": "maximize",
                    "goal_value": 100,
                },
                "command": ["${args_json}"],
            },
            entity=base_config.wandb.entity,
            project=base_config.wandb.project,
        )
    else:
        carbs = CARBS.warm_start_from_wandb(run_name=sweep_id, is_prior_observation_valid=True)

    import pprint

    pprint.pprint(params)
    sweep = wandb.controller(sweep_id)

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
                        ):
                            obs_in = ObservationInParam(
                                input={k: v["value"] for k, v in json.loads(run["config"]).items()},
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
            {k: v.value for k, v in OmegaConf.load(os.environ["WANDB_SWEEP_PARAM_PATH"]).items()}
        )
        agent_config = update_base_config(base_config, agent_config)
        train.train(config=agent_config, debug=debug, track=True)

    for _ in range(99999):
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
