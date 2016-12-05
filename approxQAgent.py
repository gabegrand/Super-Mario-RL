from qAgent import QLearningAgent
import util
import hyperparameters as hp
import numpy as np
import pickle
from datetime import datetime
import hyperparameters as hp
import features as ft

class ApproxQAgent(QLearningAgent):

    def __init__(self):
        self.weights = util.Counter()
        self.N = util.Counter()
        self.alpha = hp.ALPHA
        self.gamma = hp.GAMMA
        self.actions = hp.MAPPING.keys()
        self.iter = 0
        print self.actions

        self.features = util.Counter()
        self.prev_state = np.array([])

    def reset(self):
        self.features = util.Counter()
        self.prev_state = np.array([])

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        return self.getWeights() * self.features

    # update_dict should consist of 'state', 'action', 'nextState', and 'reward'
    def update(self, update_dict):

        state = update_dict['state']
        action = update_dict['action']
        nextState = update_dict['nextState']
        reward = update_dict['reward']

        # Update exploration function
        self.N[str(state), action] += 1

        # Ensure Mario is on the screen in both states
        if ft.marioPosition(self.prev_state) and ft.marioPosition(state):

            # Update features
            self.features = ft.getFeatures(self.prev_state, state, action)

        # Compute value of nextState
        nextStateValue = self.computeValueFromQValues(nextState)

        # Batch update weights
        new_weights = util.Counter()

        for feature in self.features:
            new_weights[feature] = self.weights[feature] + self.alpha * ((reward + self.gamma * nextStateValue) - self.getQValue(state, action)) * self.features[feature]

        self.weights = new_weights

        # Update prev state
        self.prev_state = state

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
