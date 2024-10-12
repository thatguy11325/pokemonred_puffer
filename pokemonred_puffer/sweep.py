import math
from typing import Annotated

import carbs.utils
import typer
from carbs import (
    Param,
    ParamDictType,
    ParamType,
)
from omegaconf import DictConfig, OmegaConf
from wandb_carbs import create_sweep

import wandb
from pokemonred_puffer import train

app = typer.Typer(pretty_exceptions_enable=False)


def sweep_config_to_params(sweep_config: DictConfig, prefix: str = "") -> list[Param]:
    res = []
    for k, v in sweep_config.items():
        # A little hacky. Maybe I should not make this all config based
        if k.startswith("carbs.utils"):
            param_class = getattr(carbs.utils, k.split(".")[-1])
            res += [
                Param(
                    prefix.removesuffix("-").removeprefix("-"),
                    param_class(**v),
                    (v["max"] + v["min"]) // 2
                    if v.get("is_integer", False)
                    else math.sqrt(v["max"] ** 2 + v["min"] ** 2),
                )
            ]
        elif isinstance(v, DictConfig):
            res += sweep_config_to_params(v, prefix=prefix + "-" + k)
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
):
    params = sweep_config_to_params(sweep_config)
    import pprint

    pprint.pprint(params)
    sweep_id = create_sweep(
        sweep_name=sweep_name,
        wandb_entity=base_config.wandb.entity,
        wandb_project=base_config.wandb.project,
        carb_params=params,
    )

    print(f"Beginning sweep with id {sweep_id}")
    print(f"On all nodes please run python -m pokemonred_puffer.sweep launch-agent {sweep_id}")


@app.command()
def launch_agent(
    sweep_id: str,
    base_config: Annotated[
        DictConfig, typer.Option(help="Base configuration", parser=OmegaConf.load)
    ] = "config.yaml",
):
    def _fn():
        new_config = update_base_config(base_config)
        print(new_config)
        train.train(config=new_config)

    wandb.agent(
        sweep_id=sweep_id,
        entity=base_config.wandb.entity,
        project=base_config.wandb.project,
        function=_fn,
        count=999999,
    )


if __name__ == "__main__":
    app()
