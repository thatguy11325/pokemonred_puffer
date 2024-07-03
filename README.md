# Pokemon Red (RL Edition)

This repo is designed as a library to be used for Pokemon Red RL development. It contains some convenience functions that should not be used in a library setting and should be forked. In the future, those convenience functions will be migrated so no forking is needed.

## Quickstart

### Installation

To install the library you can either

1. Clone the repo to your local machine and install it.
2. Fork the repo and clone your fork to your local machine.

For example,

```sh
pip3 install -e . 
```

### Running

After installation you can start training by running

```sh
# Run before training to test what num_envs value you should use
python3 -m pokemonred_puffer.train  --mode autotune   --yaml config.yaml --vectorization multiprocessing -w empty -r baseline.CutWithObjectRewardRequiredEventsEnv
# Default
python3 -m pokemonred_puffer.train --mode train --yaml config.yaml
# Currently used configuration
python3 -m pokemonred_puffer.train  --mode train   --yaml config.yaml --vectorization multiprocessing  -w stream_only -r baseline.CutWithObjectRewardRequiredEventsEnv
```

### Modifying for Training

So you have a run going, but you want to mess with it, what do you do?

You have a few options:

1. Start altering parameters in `config.yaml`
2. Start modifying the code directly (more on that later).
3. Use this repo as a library and make your own wrappers.

### Debugging
If you want to test your changes you can run 

```sh
python3 -m pokemonred_puffer.train --mode train --yaml config.yaml --debug
```

In emergency cases, it is advised to remove the `send_input` function calls from `environment.py` so you can test the rewards on your own schedule and not the model's.

## Directory Structure

This repo is intended to eventually be used as a library. All source files should be under the `pokemonred_puffer` directory. If you want to add a module with a `__main__`, feel free to, but under the `pokemonred_puffer` directory. Afterwards, you should be to run your main with `python3 -m pokemonred_puffer.<your-module>`

Within the `pokemonred_puffer` directory there are the following files and directories:

- `policies`: A directory for different policies to run the model with.
- `rewards`: A directory of `gym.Env` classes that keep track of rewards for a `RedGymEnv` (gym environment for Pokemon Red) object
- `wrappers`: A directory of wrappers that you may want to use, e.g. logging to the [Pokemon Red Map Visualizer](https://pwhiddy.github.io/pokerl-map-viz/).
- `cleanrl_puffer.py` - Responsible for running the actual training logic
- `environment.py` - The core logic of the Pokemon Red Gym Environment.
- `eval.py` - For generating visualizations for logging during training.
- `kanto_map_dsv.png` - A high resolution map of the Kanto region.
- `train.py` - A script and entrypoint to start training with.

## Making Changes

For simple changes, you can update `config.yaml` directly. `config.yaml` has a few important rules. For `wrappers`, `rewards` and `policies`, the wrapper, reward or policy _must_ be keyed by `module_name.class_name`. These sections can hold multiple types of `wrappers`, `rewards` or `policies`. The general layout is `label : arguments for the class`. This is until a better way with less indirection is figured out.

### Adding Wrappers

To add wrappers, add a new class that inherits from `gym.Wrapper` to the `wrappers` directory. Then update the `wrappers` section of `config.yaml`. The wrappers wrap the base environment in order, from top to bottom. The wrappers list is _not_ keyed by the class path. It is a unique name that distinguishes the collection of wrappers.

### Adding Rewards

To add rewards, add a new class to the `rewards` directory. Then update the `rewards` section of `config.yaml`. A reward section is keyed by the class path.

### Adding Policies

To add policies, add a new class to the `policies` directory. Then update the `policies` section of `config.yaml`. A policy section is keyed by the class path. It is assumed that a recurrent policy will live in the same module as the policy it wraps.

## Development

This repo uses [pre-commit](https://pre-commit.com/) to enforce formatting and linting. For development, please install this repo with:

```sh
pip3 install -e '.[dev]'
pre-commit install
```

For any changes, please submit a PR.
