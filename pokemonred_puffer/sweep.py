import json
import math
from pathlib import Path
from typing import Annotated, Any

import carbs.utils
import sweeps
import typer
import yaml
from carbs import (
    CARBS,
    CARBSParams,
    ObservationInParam,
    Param,
    ParamDictType,
    ParamType,
    WandbLoggingParams,
)

import wandb
from pokemonred_puffer import train

app = typer.Typer(pretty_exceptions_enable=False)


def sweep_config_to_params(sweep_cfg: dict[str, Any], prefix: str = "") -> list[Param]:
    res = []
    for k, v in sweep_cfg.items():
        # A little hacky. Maybe I should not make this all config based
        if k.startswith("carbs.utils"):
            param_class = getattr(carbs.utils, k.split(".")[-1])
            res += [
                Param(
                    prefix.removesuffix("/").removeprefix("/"),
                    param_class(**v),
                    (v["max"] + v["min"]) // 2
                    if v.get("is_integer", False)
                    else math.sqrt(v["max"] ** 2 + v["min"] ** 2),
                )
            ]
        elif isinstance(v, dict):
            res += sweep_config_to_params(v, prefix=prefix + "/" + k)
    return res


def update_base_config_by_key(
    base_cfg: dict[str, Any], key: str, value: ParamType
) -> dict[str, Any]:
    new_cfg = base_cfg.copy()
    keys = key.split("/", 1)
    if len(keys) == 1:
        new_cfg[keys[0]] = value
    else:
        new_cfg[keys[0]] = update_base_config_by_key(new_cfg[keys[0]], keys[1], value)
    return new_cfg


def update_base_config(base_cfg: dict[str, Any], suggestion: ParamDictType) -> dict[str, Any]:
    new_cfg = base_cfg.copy()
    for k, v in suggestion.items():
        new_cfg = update_base_config_by_key(new_cfg, k, v)
    return new_cfg


@app.command()
def launch_controller(
    base_config: Annotated[Path, typer.Option(help="Base configuration")] = Path("config.yaml"),
    sweep_config: Annotated[
        Path, typer.Option(help="CARBS sweep config. settings must match base config.")
    ] = Path("sweep-config.yaml"),
):
    with open(base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(sweep_config) as f:
        sweep_cfg = yaml.safe_load(f)
    config = CARBSParams(
        better_direction_sign=-1,
        is_wandb_logging_enabled=True,
        wandb_params=WandbLoggingParams(project_name="Pokemon", run_name="Pokemon"),
    )
    params = sweep_config_to_params(sweep_cfg)
    import pprint

    pprint.pprint(params)
    carbs = CARBS(config=config, params=params)
    sweep_id = wandb.sweep(
        sweep={"controller": {"type": "local"}, "parameters": {}},
    )
    sweep = wandb.controller(sweep_id)
    print(f"Beginning sweep with id {sweep_id}")
    print(
        f"On all nodes please run wandb with wandb.agent(sweep_id={sweep_id}, "
        "function=<your-function>)"
    )
    while not sweep.done():
        sweep_obj = sweep._sweep_object_read_from_backend()
        if sweep_obj["runs"]:
            print(sweep_obj["runs"])
            breakpoint()
            obs_in = ObservationInParam(...)  # parsed from sweep_obj. Need to figure this out
            carbs.observe(obs_in)
        suggestion = carbs.suggest()
        new_cfg = update_base_config(base_cfg, suggestion.suggestion)
        run = sweeps.SweepRun(config=new_cfg)
        sweep.schedule(run)
        sweep.print_status()


@app.command()
def launch_agent(sweep_id: str):
    wandb.agent(
        sweep_id,
        lambda params: train.train(json.dumps(yaml=params, track=True)),
        entity="Pokemon",
        project="Pokemon",
    )


if __name__ == "__main__":
    app()
