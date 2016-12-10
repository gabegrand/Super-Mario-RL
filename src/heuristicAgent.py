# An agent that chooses moves according to engineered heuristics

from datetime import datetime
import pickle
import hyperparameters as hp
import random
import numpy as np
import features as ft

class HeuristicAgent:
    def __init__(self):
        self.actions = hp.MAPPING.keys()
        self.stuck_duration = 0
        self.jumps = 0

    def getActionAndUpdate(self, state, reward):


        if self.stuck_duration > 40:
            print "Stuck!"
            if self.jumps > 20:
                self.jumps = 0
                self.stuck_duration = 0
            self.jumps += 1
            if ft.groundVertDistance(state) == 0:
                return random.choice([10, 0])
            else:
                return 10

        if ft.marioPosition(state) is not None:

            # Check if stuck
            if not ft.canMoveRight(state):
                self.stuck_duration += 1

            # Check if enemy
            if ft.distRightEnemy(state) <= 0.2 and ft.distUpEnemy(state) < 0.1 and ft.distDownEnemy(state) < 0.1:
                print "Enemy ahead!"
                if ft.groundVertDistance(state) <= 0.01:
                    print "Whoa there..."
                    return random.choice([3, 4])
                return random.choice([0, 11])

            if 0.0625 < ft.distLeftEnemy(state) <= 0.1:
                print("Enemy behind!", ft.distLeftEnemy(state))
                return random.choice([0, 11])

            if ft.groundRightDistance(state) <= 0.3:
                print "Gap ahead!"
                if ft.groundRightDistance(state) <= 0.06:
                    print "Need to jump!"
                    a = random.choice([10, 10, 10])
                    print a
                    return a
                return 9

        return np.random.choice(self.actions, 1, p=hp.PRIOR)[0]

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
        self.stuck_duration = 0

    def numStatesLearned(self):
        return None
