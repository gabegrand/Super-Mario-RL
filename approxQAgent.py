from qAgent import QLearningAgent
import util
import hyperparameters as hp
import numpy as np
import pickle
from datetime import datetime
import hyperparameters as hp
import features as feat

class ApproxQAgent(QLearningAgent):

    def __init__(self):
        self.weights = util.Counter()
        self.N = util.Counter()
        self.alpha = hp.ALPHA
        self.gamma = hp.GAMMA
        self.actions = hp.MAPPING.keys()
        self.iter = 0
        self.features = util.Counter()
        self.prev_r = None
        self.prev_a = None
        self.prev_s = None

    def getActionAndUpdate(self, state, reward):
        assert state
        assert isinstance(state, util.State)

        action_should_be_none = False

        # Only update if prev_s exists (e.g., not first iteration of action loop)
        if self.prev_s:

            # Get Q value of previous state
            prev_q = self.getQ(self.prev_s, self.prev_a)

            # Handle Mario death
            if feat.marioPosition(state.getCurr()) is None:
                print('MODEL: Mario is dead. Returning action = None.')
                action_should_be_none = True
                reward -= hp.DEATH_PENALTY
            else:
                self.features = feat.getFeatures(state)

            # Get value of state
            nextStateValue = self.computeValueFromQValues(state)

            # Batch update weights
            new_weights = util.Counter()
            for ft in self.features:
                new_weights[ft] = self.weights[ft] + self.alpha * self.getN(self.prev_s.getCurr(), self.prev_a) * (reward + self.gamma * nextStateValue - prev_q) * self.features[ft]
            self.weights = new_weights

        # Even if prev_s doesn't exist, still compute features based on curr state
        elif feat.marioPosition(state.getCurr()) is not None:
            self.features = feat.getFeatures(state)

        # If Mario is dead
        if action_should_be_none:
            action = None

        # Otherwise, compute best action to take
        else:
            action = self.computeActionFromQValues(state)

        # Store state and reward for next iteration
        self.prev_a = action
        self.prev_s = state
        self.prev_r = reward

        # Increment exploration count
        self.incN(self.prev_s.getCurr(), self.prev_a)

        return self.prev_a

    def reset(self):
        self.features = util.Counter()
        self.prev_s = None
        self.prev_a = None
        self.prev_r = None

    def getQ(self, state, action):
        return self.weights * self.features

    def numStatesLearned(self):
        return None

    def getWeights(self):
        return self.weights

    def save(self, i, j, diagnostics):

        # Build save file name
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD + '-iter-' + str(i + j)

        saved_vals = {'weights': self.weights, 'N': self.N, 'diagnostics': diagnostics}

        with open('save/' + fname + '.pickle', 'wb') as handle:
            pickle.dump(saved_vals, handle)

    def load(self, fname):
        try:
            with open('save/' + fname, 'rb') as handle:
                saved_vals = pickle.load(handle)
                self.weights = saved_vals['weights']
                self.N = saved_vals['N']
        except:
            ValueError('Failed to load file %s' % ('save/' + fname))
