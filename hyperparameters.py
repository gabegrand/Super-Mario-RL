TRAINING_ITERATIONS = 10

WORLD = '1-1'
LEVEL = 'ppaquette/SuperMarioBros-' + WORLD + '-Tiles-v0'

# Q Learning Agent Parameters
ALPHA = 0.5     # Learning rate
EPSILON = 0.2   # Radom move probability
GAMMA = 0.8     # Discount factor

# Penalty for dying in reward function
DEATH_PENALTY = 100

# Percentage of score increase added to reward
SCORE_FACTOR = 1

# Action mapping
mapping = {
    0: [0, 0, 0, 0, 0, 0],  # NOOP
    1: [1, 0, 0, 0, 0, 0],  # Up
    2: [0, 0, 1, 0, 0, 0],  # Down
    3: [0, 1, 0, 0, 0, 0],  # Left
    4: [0, 1, 0, 0, 1, 0],  # Left + A
    5: [0, 1, 0, 0, 0, 1],  # Left + B
    6: [0, 1, 0, 0, 1, 1],  # Left + A + B
    7: [0, 0, 0, 1, 0, 0],  # Right
    8: [0, 0, 0, 1, 1, 0],  # Right + A
    9: [0, 0, 0, 1, 0, 1],  # Right + B
    10: [0, 0, 0, 1, 1, 1],  # Right + A + B
    11: [0, 0, 0, 0, 1, 0],  # A
    12: [0, 0, 0, 0, 0, 1],  # B
    13: [0, 0, 0, 0, 1, 1],  # A + B
}
