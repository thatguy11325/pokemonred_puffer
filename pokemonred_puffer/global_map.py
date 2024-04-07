import os
import json

MAP_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")
GLOBAL_MAP_SHAPE = (444, 436)

with open(MAP_PATH) as map_data:
    MAP_DATA = json.load(map_data)["regions"]
MAP_DATA = {int(e["id"]): e for e in MAP_DATA}


# Handle KeyErrors
def local_to_global(r: int, c: int, map_n: int):
    try:
        (
            map_x,
            map_y,
        ) = MAP_DATA[map_n]["coordinates"]
        gy = r + map_y
        gx = c + map_x
        if 0 > gy >= GLOBAL_MAP_SHAPE[0] or 0 > gx >= GLOBAL_MAP_SHAPE[1]:
            print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
            return 0, 0
        return gy, gx
    except KeyError:
        print(f"Map id {map_n} not found in map_data.json.")
        return r + 0, c + 0
