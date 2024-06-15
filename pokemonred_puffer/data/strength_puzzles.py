STRENGTH_SOLUTIONS = {}
# Seafoam 1F Left
STRENGTH_SOLUTIONS[(63, 14, 22, 18, 11, 192)] = [
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "LEFT",
]
STRENGTH_SOLUTIONS[(63, 14, 22, 19, 10, 192)] = ["DOWN", "LEFT"] + STRENGTH_SOLUTIONS[
    (63, 14, 22, 18, 11, 192)
]
STRENGTH_SOLUTIONS[(63, 14, 22, 18, 9, 192)] = ["RIGHT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 14, 22, 19, 10, 192)
]
STRENGTH_SOLUTIONS[(63, 14, 22, 17, 10, 192)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 14, 22, 18, 9, 192)
]

# Seafoam 1F right
STRENGTH_SOLUTIONS[(63, 11, 30, 26, 8, 192)] = [
    "UP",
    "RIGHT",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
]
STRENGTH_SOLUTIONS[(63, 11, 30, 27, 7, 192)] = ["DOWN", "LEFT"] + STRENGTH_SOLUTIONS[
    (63, 11, 30, 26, 8, 192)
]
STRENGTH_SOLUTIONS[(63, 11, 30, 26, 6, 192)] = ["RIGHT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 11, 30, 27, 7, 192)
]
STRENGTH_SOLUTIONS[(63, 11, 30, 25, 7, 192)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 11, 30, 26, 6, 192)
]

# Seafoam B1 left

STRENGTH_SOLUTIONS[(63, 10, 21, 16, 6, 159)] = ["RIGHT"]
STRENGTH_SOLUTIONS[(63, 10, 21, 17, 5, 159)] = ["LEFT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 10, 21, 16, 6, 159)
]
STRENGTH_SOLUTIONS[(63, 10, 21, 17, 7, 159)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 10, 21, 16, 6, 159)
]

# Seafoam B1 right

STRENGTH_SOLUTIONS[(63, 10, 26, 21, 6, 159)] = ["RIGHT"]
STRENGTH_SOLUTIONS[(63, 10, 26, 22, 5, 159)] = ["LEFT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 10, 26, 21, 6, 159)
]
STRENGTH_SOLUTIONS[(63, 10, 26, 22, 7, 159)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 10, 26, 21, 6, 159)
]

# Seafoam B2 left

STRENGTH_SOLUTIONS[(63, 10, 22, 17, 6, 160)] = ["RIGHT"]
STRENGTH_SOLUTIONS[(63, 10, 22, 18, 5, 160)] = ["LEFT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 10, 22, 17, 6, 160)
]
STRENGTH_SOLUTIONS[(63, 10, 22, 18, 7, 160)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 10, 22, 17, 6, 160)
]

# Seafoam B2 right

STRENGTH_SOLUTIONS[(63, 10, 27, 24, 6, 160)] = ["LEFT"]
STRENGTH_SOLUTIONS[(63, 10, 27, 23, 7, 160)] = ["RIGHT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 10, 27, 24, 6, 160)
]

# We skip seafoam b3 since that is for articuno
# TODO: Articuno
