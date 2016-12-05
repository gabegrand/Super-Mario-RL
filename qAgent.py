import util
import random
import numpy as np
import pickle
from datetime import datetime
import hyperparameters as hp
import features as ft
import math

class QLearningAgent:

    def __init__(self):
        self.Q = util.Counter()
        self.N = util.Counter() # visit count
        self.alpha = hp.ALPHA
        self.iter = 0
        self.gamma = hp.GAMMA
        self.actions = hp.MAPPING.keys()

    def reset(self):
        return None

    def getNValue(self, state, action):
        return self.N[str(state), action]

    def getQValue(self, state, action):
        return self.Q[str(state), action]

    def computeValueFromQValues(self, state):

        # Keep track of values of each action
        action_values = util.Counter()

        # Get value of each action
        for action in self.actions:
            # avoid dividing by zero by adding 1
            action_values[action] = self.getQValue(state, action) + hp.K / (self.getNValue(state, action) + 1.0)

        # Return max value
        return action_values[action_values.argMax()]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
        """

        # Keep track of values of each action
        action_values = []

        # Get value of each action
        for action in self.actions:
            action_values.append(self.getQValue(state, action))

        # Compute max value over all actions
        max_value = max(action_values)

        # Get indices of all actions that lead to max value
        indices = [i for i, x in enumerate(action_values) if x == max_value]

        # Return action with max value, breaking ties randomly
        action = self.actions[random.choice(indices)]

        return action

    def getAction(self, state):

        # With probability epsilon, choose random action
        if util.flipCoin(max(hp.MIN_EPSILON, 5.0 / (5.0 + self.iter))):
            action = np.random.choice(self.actions, 1, p=hp.PRIOR)[0]
            self.iter += 1

        # With probability 1 - epsilon, choose best action according to Q values
        else:
            action = self.computeActionFromQValues(state)

        return action

    # update vals should consist of state, action, nextState, and reward
    def update(self, update_dict):

        state = update_dict['state']
        action = update_dict['action']
        nextState = update_dict['nextState']
        reward = update_dict['reward']

        self.N[str(state), action] += 1

        # Compute value of nextState
        nextStateValue = self.computeValueFromQValues(nextState)

        # Update Q value with running average based on observed sample
        self.Q[str(state), action] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.gamma * nextStateValue)

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

class ApproxSarsaAgent(ApproxQAgent):

    # only method that is different
    # update_dict must consist of state, action, nextState, nextAction, reward
    def update(self, update_dict):

        state = update_dict['state']
        action = update_dict['action']
        nextState = update_dict['nextState']
        nextAction = update_dict['nextAction']
        reward = update_dict['reward']

        self.N[str(state), action] += 1

        # Ensure Mario is on the screen in both states
        # if ft.marioPosition(state) and ft.marioPosition(nextState):
        if ft.marioPosition(nextState):

            # Update features
            features = ft.getFeatures(nextState, action)
            self.features = features
        else:
            print "update: Mario not on screen"

        # Update prev state
        self.prev_state = state

        # Compute value of nextState SARSA style
        nextStateValue = self.getQValue(state, nextAction)

        # Batch update weights
        new_weights = util.Counter()

        for feature in self.features:
            new_weights[feature] = self.weights[feature] + self.alpha * ((reward + self.gamma * nextStateValue) - self.getQValue(state, action)) * self.features[feature]

        self.weights = new_weights
        print self.weights
