# Game
WORLD = '1-4'
LEVEL = 'ppaquette/SuperMarioBros-' + WORLD + '-Tiles-v0'

TRAINING_ITERATIONS = 100

# Whether to load from existing Q values file
LOAD_FROM = '2016-11-23-18-58-world-1-4-iter-91.pickle'

# How often to save Q values
SAVE_EVERY = 10

# 0 is Q-Learning, should 1 be approximate Q?
AGENT_TYPE = 0

# Q Learning Agent Parameters
ALPHA = 0.3     # Learning rate
EPSILON = 0.2   # Random move probability
GAMMA = 0.9     # Discount factor

# Penalty for dying in reward function
DEATH_PENALTY = 100

# Proportion of score increase added to reward (0 to 1)
SCORE_FACTOR = 1

# Action mapping
MAPPING = {
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
