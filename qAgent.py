import util
import random
import numpy as np
import pickle
from datetime import datetime
import hyperparameters as hp
import features as ft

class QLearningAgent:

    def __init__(self):
        self.Q = util.Counter()
        self.N = util.Counter() # visit count
        self.alpha = hp.ALPHA
        self.iter = 0
        self.gamma = hp.GAMMA
        self.actions = hp.MAPPING.keys()

        self.prev_r = None
        self.prev_a = None
        self.prev_s = None

    def getActionAndUpdate(self, state, reward):
        assert state is not None
        assert isinstance(state, np.ndarray)

        # TODO can state ever be terminal?
        if self.prev_s is not None:
            nextStateValue = self.computeValueFromQValues(state)
            prev_q = self.getQ(self.prev_s, self.prev_a)
            self.setQ(str(self.prev_s), self.prev_a,
                prev_q + self.alpha * self.N[str(self.prev_s), self.prev_a] * (reward + self.gamma * nextStateValue - prev_q))

        self.prev_a = self.computeActionFromQValues(state)
        self.prev_s = state
        self.prev_r = reward

        self.incN(self.prev_s, self.prev_a)
        return self.prev_a

    def reset(self):
        return None

    def getN(self, state, action):
        return self.N[str(state), action]

    def incN(self, state, action):
        self.N[str(state), action] += 1

    def getQ(self, state, action):
        if hasattr(self, 'Q'):
            return self.Q[str(state), action]
        else:
            return self.weights * self.features

    def setQ(self, state, action, q_val):
        self.Q[str(state), action] = q_val

    def computeValueFromQValues(self, state):

        # Keep track of values of each action
        action_values = util.Counter()

        # Get value of each action
        for action in self.actions:
            # avoid dividing by zero by adding 1
            action_values[action] = self.getQ(state, action)

        # Return max value
        return action_values[action_values.argMax()]

    def computeActionFromQValues(self, state):

        if util.flipCoin(max(hp.MIN_EPSILON, 5.0 / (5.0 + self.iter))):
            self.iter += 1
            return np.random.choice(self.actions, 1, p=hp.PRIOR)[0]
        """
          Compute the best action to take in a state.
        """

        # Keep track of values of each action
        action_values = []

        # Get value of each action
        for action in self.actions:
            action_values.append(self.getQ(state, action) + hp.K / (self.getN(state, action) + 1.0))

        # Compute max value over all actions
        max_value = max(action_values)

        # Get indices of all actions that lead to max value
        indices = [i for i, x in enumerate(action_values) if x == max_value]

        # Return action with max value, breaking ties randomly
        action = self.actions[random.choice(indices)]

        return action

    def numStatesLearned(self):
        return len(self.Q.keys())

    def save(self, i, j, diagnostics):

        # Build save file name
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD + '-iter-' + str(i + j)

        saved_vals = {'Q': self.Q, 'N': self.N, 'diagnostics': diagnostics}

        with open('save/' + fname + '.pickle', 'wb') as handle:
            pickle.dump(saved_vals, handle)

    def load(self, fname):
        try:
            with open('save/' + fname, 'rb') as handle:
                saved_vals = pickle.load(handle)
                self.Q = saved_vals['Q']
                self.N = saved_vals['N']
        except:
            ValueError('Failed to load file %s' % ('save/' + fname))