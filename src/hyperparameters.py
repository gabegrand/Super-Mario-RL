from collections import OrderedDict
import math

# Agents
from qAgent import QLearningAgent
from approxQAgent import ApproxQAgent
from approxSarsaAgent import ApproxSarsaAgent
from randomAgent import RandomAgent
from heuristicAgent import HeuristicAgent

# Whether to display certain warnings used for testing
DISPLAY_WARNINGS = False

# Game
WORLD = (1, 1)
WORLD_STR = '-'.join([str(x) for x in WORLD])
LEVEL = 'ppaquette/SuperMarioBros-' + WORLD_STR + '-Tiles-v0'

TRAINING_ITERATIONS = 100

# Whether to load from existing Q values file
LOAD_FROM = None

# How often to save desired values
SAVE_EVERY = 10

# Which agent to use
AGENT_TYPE = ApproxQAgent

# Q Learning Agent Parameters
ALPHA = 0.01          # Learning rate
MIN_EPSILON = 0.05   # Random move probability
GAMMA = 0.95         # Discount factor
LAMBDA = 0.8         # Eligibility trace decay in Q(LAMBDA)
MIN_LAMBDA = 0.1    # Minimum discounted value for which weight updates get computed

# Note: when selecting LAMBDA params, we require LAMBDA^x > MIN_LAMBDA, so that
# x is the number of previous states for which the weight update is applied
MAX_TRACES = max(int(math.log(MIN_LAMBDA, LAMBDA)), 1)
print('Eligibility trace length: %d' % MAX_TRACES)

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

# Distance to castle for every level
WIN_DISTANCES = OrderedDict([((1, 1), 3266), ((1, 2), 3266), ((1, 3), 2514), ((1, 4), 2430),
                             ((2, 1), 3298), ((2, 2), 3266), ((2, 3), 3682), ((2, 4), 2430),
                             ((3, 1), 3298), ((3, 2), 3442), ((3, 3), 2498), ((3, 4), 2430),
                             ((4, 1), 3698), ((4, 2), 3266), ((4, 3), 2434), ((4, 4), 2942),
                             ((5, 1), 3282), ((5, 2), 3298), ((5, 3), 2514), ((5, 4), 2429),
                             ((6, 1), 3106), ((6, 2), 3554), ((6, 3), 2754), ((6, 4), 2429),
                             ((7, 1), 2962), ((7, 2), 3266), ((7, 3), 3682), ((7, 4), 3453),
                             ((8, 1), 6114), ((8, 2), 3554), ((8, 3), 3554), ((8, 4), 4989)])

# The flagpole is 40 meters before the castle
LEVEL_WIN_DIST = WIN_DISTANCES[WORLD] - 40
