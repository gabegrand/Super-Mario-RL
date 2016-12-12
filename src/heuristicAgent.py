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

        state = state.getTiles()
        mpos = ft.marioPosition(state)

        action = None

        # Get Mario unstuck
        if self.stuck_duration > 30:
            print "Stuck!"
            if self.jumps > 20:
                self.jumps = 0
                self.stuck_duration = 0
            self.jumps += 1
            if ft.groundVertDistance(state, mpos) == 0:
                action = random.choice([10, 0])
            else:
                action = 10

        elif mpos is not None:

            # Check if stuck
            if not ft.canMoveRight(state, mpos):
                self.stuck_duration += 1

            # Check if enemy left
            if ft.distRightEnemy(state, mpos) <= 0.2 and ft.distUpEnemy(state, mpos) < 0.1 and ft.distDownEnemy(state, mpos) < 0.2:
                print "Enemy ahead!"
                action = random.choice([0, 11])

            # Check if enemy right
            elif 0.0625 < ft.distLeftEnemy(state, mpos) <= 0.1:
                print("Enemy behind!", ft.distLeftEnemy(state, mpos))
                action = random.choice([0, 11])

            # Check if gap
            elif ft.groundRightDistance(state, mpos) <= 0.3:
                print "Gap ahead!"
                if ft.groundRightDistance(state, mpos) <= 0.06:
                    print "Need to jump!"
                    action = 10
                else:
                    action = 9

        random_action = np.random.choice(self.actions, 1, p=hp.PRIOR)[0]

        if not action:
            return random_action
        else:
            return np.random.choice([action, random_action], 1, p=[0.99, 0.01])[0]

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
