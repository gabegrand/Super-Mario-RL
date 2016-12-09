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

        # TODO can state ever be terminal?
        if self.prev_s:
            self.features = feat.getFeatures(state)

            # Batch update weights
            new_weights = util.Counter()

            nextStateValue = self.computeValueFromQValues(state)
            for ft in self.features:
                new_weights[ft] = self.weights[ft] + self.alpha * self.getN(self.prev_s.getCurr(), self.prev_a) * (reward + self.gamma * nextStateValue - self.weights[ft])
            self.weights = new_weights

        self.prev_a = self.computeActionFromQValues(state)
        self.prev_s = state
        self.prev_r = reward

        self.incN(self.prev_s.getCurr(), self.prev_a)
        return self.prev_a

    def reset(self):
        self.features = util.Counter()
        self.prev_s = None
        self.prev_a = None
        self.prev_r = None

    def getQValue(self, state, action):
        return self.weights * self.features

    def numStatesLearned(self):
        return None

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
