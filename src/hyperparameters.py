# Whether to display certain warnings used for testing
DISPLAY_WARNINGS = False

# Game
WORLD = '1-1'
LEVEL = 'ppaquette/SuperMarioBros-' + WORLD + '-Tiles-v0'

TRAINING_ITERATIONS = 500

# Whether to load from existing Q values file
LOAD_FROM = None

# How often to save desired values
SAVE_EVERY = 10

# 0 is Random Agent, 1 regular QLearningAgent, 2 is ApproxQAgent, 3 is ApproxSarsaAgent
AGENT_TYPE = 2

# Q Learning Agent Parameters
ALPHA = 0.1     # Learning rate
MIN_EPSILON = 0.05   # Random move probability
GAMMA = 0.95     # Discount factor

EP_DEC = 500.0

K = 10.0 # k value for exploration function, should be a float; set to 0 to ignore

# Penalty for dying in reward function
DEATH_PENALTY = 100

# Proportion of score increase added to reward (0 to 1)
SCORE_FACTOR = 1.0

# How many frames Mario is stuck for before the model rescues him
STUCK_DURATION = 80

# How many jumps the model uses to get Mario unstuck
MAX_JUMPS = 25

# Action mapping
MAPPING = {
    # 0: [0, 0, 0, 0, 0, 0],  # NOOP
    # 1: [1, 0, 0, 0, 0, 0],  # Up
    # 2: [0, 0, 1, 0, 0, 0],  # Down
    3: [0, 1, 0, 0, 0, 0],  # Left
    4: [0, 1, 0, 0, 1, 0],  # Left + A
    5: [0, 1, 0, 0, 0, 1],  # Left + B
    6: [0, 1, 0, 0, 1, 1],  # Left + A + B
    7: [0, 0, 0, 1, 0, 0],  # Right
    8: [0, 0, 0, 1, 1, 0],  # Right + A
    9: [0, 0, 0, 1, 0, 1],  # Right + B
    10: [0, 0, 0, 1, 1, 1],  # Right + A + B
    11: [0, 0, 0, 0, 1, 0],  # A
    # 12: [0, 0, 0, 0, 0, 1],  # B
    # 13: [0, 0, 0, 0, 1, 1],  # A + B
}

# needs to be float
NORM = 50.0

# prior dist on actions
PRIOR = [
    2/NORM,    #3
    2/NORM,    #4
    2/NORM,    #5
    2/NORM,    #6
    5/NORM,    #7
    5/NORM,   #8
    20/NORM,    #9
    10/NORM,   #10
    2/NORM,    #11
    ]
