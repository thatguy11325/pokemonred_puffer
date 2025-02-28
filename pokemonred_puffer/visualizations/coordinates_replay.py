import argparse
import os

import cv2
import mediapy
import numpy as np
from tqdm import tqdm

from pokemonred_puffer.eval import BACKGROUND
from pokemonred_puffer.global_map import local_to_global

PLAYER_PATH = os.path.join(os.path.dirname(__file__), "player.png")
PLAYER = np.array(cv2.imread(PLAYER_PATH, cv2.IMREAD_COLOR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coords_folder")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    # steps is a list of lists where the index maps to the step count
    steps = []
    # load data
    for path in os.listdir(args.coords_folder):
        with open(os.path.join(args.coords_folder, path)) as f:
            for i, line in enumerate(f):
                map_n, y_pos, x_pos = line.strip(" \n").split(",")
                if i == len(steps):
                    steps.append([])
                steps[i].append((int(map_n), int(y_pos), int(x_pos)))

    with mediapy.VideoWriter(
        os.path.join(args.output_dir, "coords.mp4"), BACKGROUND.shape[:2], fps=60
    ) as writer:
        for step in tqdm(steps):
            frame = BACKGROUND.copy()
            for map_n, y_pos, x_pos in step:
                y, x = local_to_global(y_pos, x_pos, map_n)
                y *= 16
                x *= 16
                frame[y : y + PLAYER.shape[0], x : x + PLAYER.shape[1]] = PLAYER
            writer.add_image(frame)


if __name__ == "__main__":
    main()
