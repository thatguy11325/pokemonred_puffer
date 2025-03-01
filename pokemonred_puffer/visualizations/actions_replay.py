import argparse
import os

import mediapy
from omegaconf import OmegaConf
from tqdm import tqdm

from pokemonred_puffer.rewards.baseline import ObjectRewardRequiredEventsMapIdsFieldMoves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rom_path")
    parser.add_argument("actions_folder")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    # steps is a list of lists where the index maps to the step count
    os.makedirs(args.output_dir, exist_ok=True)
    for path in os.listdir(str(args.actions_folder)):
        if not path.endswith("actions.csv"):
            continue
        env_id, _ = path.split("-")
        # The config must match what was used for training
        with (
            open(os.path.join(args.actions_folder, path)) as f,
            mediapy.VideoWriter(
                os.path.join(args.output_dir, f"actions-{env_id}.mp4"), (144, 160), fps=60
            ) as writer,
        ):
            config = OmegaConf.load("config.yaml")
            config.env.gb_path = args.rom_path
            config.env.log_frequency = None
            env = ObjectRewardRequiredEventsMapIdsFieldMoves(
                config.env,
                config.rewards.get("baseline.ObjectRewardRequiredEventsMapIdsFieldMoves").reward,
            )
            env.reset()
            writer.add_image(env.render()[:, :])
            # Read lines so we can get an estimate of the line count
            actions = f.readlines()
            for action in tqdm(actions):
                env.step(int(action.strip()))
                writer.add_image(env.render()[:, :])


if __name__ == "__main__":
    main()
