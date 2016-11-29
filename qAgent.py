import util
import random
import numpy as np
import pickle
from datetime import datetime
import hyperparameters as hp

class QLearningAgent:

    def __init__(self):
        self.Q = util.Counter()
        self.alpha = hp.ALPHA
        self.epsilon = hp.EPSILON
        self.gamma = hp.GAMMA
        self.actions = hp.MAPPING.keys()

    def getQValue(self, state, action):
        return self.Q[state, action]

    def computeValueFromQValues(self, state):

        # Keep track of values of each action
        action_values = util.Counter()

        # Get value of each action
        for action in self.actions:
            action_values[action] = self.getQValue(state, action)

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

        # print "Action: %d, Value: %f" % (action, max_value)

        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
        """

        # With probability epsilon, choose random action
        if util.flipCoin(self.epsilon):
            action = random.choice(self.actions)

        # With probability 1 - epsilon, choose best action according to Q values
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):

        # Compute value of nextState
        nextStateValue = self.computeValueFromQValues(nextState)

        # Update Q value with running average based on observed sample
        self.Q[state, action] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.gamma * nextStateValue)

    def numStatesLearned(self):
        return len(self.Q.keys())

    def save(self, i, j):

        # Build save file name
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD + '-iter-' + str(i + j)

        with open('save/' + fname + '.pickle', 'wb') as handle:
            pickle.dump(self.Q, handle)

    def load(self, fname):
        try:
            with open('save/' + fname, 'rb') as handle:
                self.Q = pickle.load(handle)
        except:
            ValueError('Failed to load file %s' % ('save/' + fname))

class ApproxQAgent(QLearningAgent):

    def __init__(self):
        self.weights = util.Counter()
        self.alpha = hp.ALPHA
        self.epsilon = hp.EPSILON
        self.gamma = hp.GAMMA
        self.actions = hp.MAPPING.keys()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        return self.getWeights() * self.featExtractor.getFeatures(state, action)

    def update(self, features, action, nextState, reward):
        # Compute value of nextState
        nextStateValue = self.computeValueFromQValues(nextState)

        # Get dictionary (actually a util.Counter object) of features and values
        feature_dict = self.featExtractor.getFeatures(state, action)

        # Update each weight iteratively based on feature
        for feature in feature_dict:
            self.weights[feature] = self.weights[feature] + self.alpha * ((reward + self.discount * nextStateValue) - self.getQValue(state, action)) * feature_dict[feature]
