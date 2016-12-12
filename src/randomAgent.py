# An agent that chooses moves randomly

from datetime import datetime
import pickle
import hyperparameters as hp
import random
import numpy as np
import features as feat

# Sample actions uniformly at random
class RandomAgent:
    def __init__(self):
        self.actions = hp.MAPPING.keys()

    def getActionAndUpdate(self, state, reward):
        return np.random.choice(self.actions, 1)[0]

    def save(self, i, j, diagnostics):
        # Build save file name
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD_STR + '-iter-' + str(i + j)

        saved_vals = {'diagnostics': diagnostics}

        with open('save/' + fname + '.pickle', 'wb') as handle:
            pickle.dump(saved_vals, handle)

    # Dummy methods
    def load(self, fname):
        return None

    def reset(self):
        self.stuck_duration = 0

    def numStatesLearned(self):
        return None

# Sample actions from weighted prior
class WeightedRandomAgent(RandomAgent):

    def __init__(self):
        self.actions = hp.MAPPING.keys()
        self.stuck_duration = 0
        self.jumps = 0

    def getActionAndUpdate(self, state, reward):

        action = np.random.choice(self.actions, 1, p=hp.PRIOR)[0]

        # If Mario is stuck, overwrite action with jump
        if state is not None:
            mpos = feat.marioPosition(state.getTiles())

            if feat.stuck(state.getTiles(), mpos):
                self.stuck_duration += 1

                # If stuck for too long, rescue him
                if self.stuck_duration > hp.STUCK_DURATION:
                    print "MODEL: Mario is stuck. Forcing jump to rescue..."

                    # On ground, get started with jump
                    if feat.groundVertDistance(state.getTiles(), mpos) == 0:
                        action = random.choice([0, 10])
                    # Jump!
                    else:
                        action = 10
                        self.jumps += 1

                    # Stop jumping and reset
                    if self.jumps > hp.MAX_JUMPS:
                        self.jumps = 0
                        self.stuck_duration = 0

        return action
