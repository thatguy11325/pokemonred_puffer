import argparse
from datetime import datetime, timedelta
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
    parser.add_argument("coords_dir")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride when reading the coordinates array"
        " or how to bucket coordinates by timestamp in seconds in seconds mode.",
    )
    parser.add_argument(
        "--left-crop",
        type=int,
        default=0,
        help="Amount of steps or seconds relative to the "
        "earliest timestamp detected to to crop from the left side.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Amount of steps or seconds to read not including stride. 0 means read all.",
    )
    parser.add_argument(
        "--downsample", type=int, default=1, help="Amount to downsample the video by. Max is 16"
    )
    parser.add_argument(
        "--image-crop",
        default=[0, 0, 0, 0],
        type=lambda x: [int(y) for y in x.split(",")],
        help="top,vlength,left,hlength cropping in pixels off the original image",
    )
    parser.add_argument(
        "--sync-method",
        choices=["time", "steps"],
        default="steps",
        help="How to synchronize the coordinates files",
    )
    parser.add_argument("--coords-file", help="Only render this file in coords_dir.")
    args = parser.parse_args()

    # steps is a list of lists where the index maps to the step count
    steps: dict[int, set[tuple[int, int, int]]] = {}
    # load data
    earliest_time: None | datetime = None
    if args.sync_method == "steps":
        left_crop = args.left_crop
        length = args.length

    files = []
    for path in os.listdir(str(args.coords_dir)):
        if args.coords_file and path != args.coords_file:
            continue
        if not path.endswith("coords.csv"):
            continue
        if not len(path.split("-")) == 3:
            continue
        files.append(path)

        if args.sync_method == "time":
            for path in files:
                date_string = path.split("-")[1]
                ts = datetime.strptime(date_string, "%Y%m%d%H%M%S")
                earliest_time = min(earliest_time or ts, ts)
            assert earliest_time
            left_crop = int((earliest_time + timedelta(seconds=args.left_crop)).timestamp())
            length = args.length

    for path in tqdm(files):
        if args.coords_file and path != args.coords_file:
            continue
        if not path.endswith("coords.csv"):
            continue
        with open(os.path.join(args.coords_dir, path)) as f:
            for i, line in enumerate(f):
                timestamp, map_n, y_pos, x_pos = line.strip(" \n").split(",")
                if args.sync_method == steps:
                    if i < left_crop:
                        continue
                    if (i - left_crop) % args.stride != 0:
                        continue
                    if i > (left_crop + length):
                        break
                    if i >= len(steps):
                        steps.append(set([]))
                    steps[i].add((int(map_n), int(y_pos), int(x_pos)))
                else:
                    timestamp = int(datetime.strptime(timestamp, "%Y%m%d%H%M%S").timestamp())
                    if timestamp < left_crop:
                        continue
                    if timestamp % args.stride != 0:
                        continue
                    if (timestamp - left_crop) > args.length:
                        break

                    key = int(timestamp) // args.stride
                    if key not in steps:
                        steps[key] = set([])
                    steps[key].add((int(map_n), int(y_pos), int(x_pos)))

    sorted_steps = sorted(steps.items(), key=lambda k: k[0])

    top, vlength, left, hlength = args.image_crop
    if not vlength:
        vlength = BACKGROUND.shape[0] - top
    if not hlength:
        hlength = BACKGROUND.shape[1] - left
    background = BACKGROUND.copy()[
        top : top + vlength : args.downsample, left : left + hlength : args.downsample
    ]
    player = PLAYER.copy()[:: args.downsample, :: args.downsample]
    with mediapy.VideoWriter(
        os.path.join(args.output_dir, "coords.mp4"),
        background.shape[:2],
        fps=24,
    ) as writer:
        for _, step in tqdm(sorted_steps):
            # This is slow. See if we can make all the frames in a threadpool
            frame = background.copy()
            for map_n, y_pos, x_pos in step:
                y, x = local_to_global(y_pos, x_pos, map_n)
                y *= 16
                x *= 16
                y = (y - top) // args.downsample
                x = (x - left) // args.downsample
                # check if the image is in frame
                if (
                    0 <= y < y + player.shape[0] < background.shape[0]
                    and 0 <= x < x + player.shape[1] < background.shape[1]
                ):
                    frame[
                        y : y + player.shape[0],
                        x : x + player.shape[1],
                    ] = player
            writer.add_image(frame)
    os.sync()


if __name__ == "__main__":
    main()
