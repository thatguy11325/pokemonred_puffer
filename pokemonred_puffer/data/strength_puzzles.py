STRENGTH_SOLUTIONS = {}

###################
# SEAFOAM ISLANDS #
###################

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

################
# VICTORY ROAD #
################

# 1F Switch 1
STRENGTH_SOLUTIONS[(63, 19, 9, 5, 14, 108)] = [
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "DOWN",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "DOWN",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "UP",
    "UP",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "RIGHT",
    "DOWN",
]

STRENGTH_SOLUTIONS[(63, 19, 9, 4, 15, 108)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 19, 9, 5, 14, 108)
]
STRENGTH_SOLUTIONS[(63, 19, 9, 5, 16, 108)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 19, 9, 4, 15, 108)
]

# 2F Switch 1
STRENGTH_SOLUTIONS[(63, 18, 8, 5, 14, 194)] = [
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "RIGHT",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
]

STRENGTH_SOLUTIONS[(63, 18, 8, 4, 13, 194)] = ["RIGHT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 18, 8, 5, 14, 194)
]
STRENGTH_SOLUTIONS[(63, 18, 8, 3, 14, 194)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 18, 8, 4, 13, 194)
]
STRENGTH_SOLUTIONS[(63, 18, 8, 4, 15, 194)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 18, 8, 3, 14, 194)
]

# 3F Switch 3
STRENGTH_SOLUTIONS[(63, 19, 26, 22, 4, 198)] = [
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "DOWN",
    "DOWN",
    "RIGHT",
    "DOWN",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "UP",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "RIGHT",
    "RIGHT",
]

STRENGTH_SOLUTIONS[(63, 19, 26, 23, 3, 198)] = ["DOWN", "LEFT"] + STRENGTH_SOLUTIONS[
    (63, 19, 26, 22, 4, 198)
]
STRENGTH_SOLUTIONS[(63, 19, 26, 22, 2, 198)] = ["RIGHT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 19, 26, 23, 3, 198)
]
STRENGTH_SOLUTIONS[(63, 19, 26, 21, 3, 198)] = ["UP", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 19, 26, 22, 2, 198)
]

# 3F Boulder in hole
STRENGTH_SOLUTIONS[(63, 16, 17, 21, 15, 198)] = ["RIGHT", "RIGHT", "RIGHT"]
STRENGTH_SOLUTIONS[(63, 16, 17, 22, 16, 198)] = ["LEFT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 16, 17, 21, 15, 198)
]
STRENGTH_SOLUTIONS[(63, 16, 17, 22, 14, 198)] = ["LEFT", "DOWN"] + STRENGTH_SOLUTIONS[
    (63, 16, 17, 21, 15, 198)
]


# 2F final switch
STRENGTH_SOLUTIONS[(63, 20, 27, 24, 16, 194)] = [
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
]

STRENGTH_SOLUTIONS[(63, 20, 27, 23, 17, 194)] = ["RIGHT", "UP"] + STRENGTH_SOLUTIONS[
    (63, 20, 27, 24, 16, 194)
]
STRENGTH_SOLUTIONS[(63, 20, 27, 22, 16, 194)] = ["DOWN", "RIGHT"] + STRENGTH_SOLUTIONS[
    (63, 20, 27, 23, 17, 194)
]
