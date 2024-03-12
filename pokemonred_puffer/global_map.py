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
        return r + map_y, c + map_x
    except KeyError:
        print(f"Map id {map_n} not found in map_data.json.")
        return r + 0, c + 0
