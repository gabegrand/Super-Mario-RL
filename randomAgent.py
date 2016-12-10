# An agent that chooses moves randomly

from datetime import datetime
import pickle
import hyperparameters as hp
import random

class RandomAgent:
    def __init__(self):
        self.actions = hp.MAPPING.keys()

    def getActionAndUpdate(self, state, reward):
        return random.choice(self.actions)

    def save(self, i, j, diagnostics):
        # Build save file name
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD + '-iter-' + str(i + j)

        saved_vals = {'diagnostics': diagnostics}

        with open('save/' + fname + '.pickle', 'wb') as handle:
            pickle.dump(saved_vals, handle)

    # Dummy methods
    def load(self, fname):
        return None

    def reset(self):
        return None

    def numStatesLearned(self):
        return None