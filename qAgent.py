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
        self.alpha = hp.ALPHA
        self.epsilon = hp.EPSILON
        self.gamma = hp.GAMMA
        self.actions = hp.MAPPING.keys()

    def getQValue(self, state, action):

        # Convert numpy array to string
        state = str(state)

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

        # Convert numpy array to string
        state = str(state)

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
        self.features = util.Counter()
        self.last_state = None
        # self.prev_state = None

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):

        # Handle initial case where there is no previous state
        # if self.prev_state is None:
        #     print "getQValue: Prev state is none; returning 0"
        #     return 0

        # else:
        # Ensure Mario is on the screen in both states
        # if ft.marioPosition(self.prev_state) and ft.marioPosition(state):
        if ft.marioPosition(state):

            return self.getWeights() * ft.getFeatures(state, action)
            self.last_state = state
            return w * f

        # If Mario not on screen, calculate Q based on last known state
        elif self.last_state is not None:
            w = self.getWeights() * ft.getFeatures(self.last_state, action)

        # If there is no last known state, just return Q = 0
        else:
            print "getQValue: Mario not on screen and no cached last state. Returning 0."
            return 0

    def update(self, state, action, nextState, reward):

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

        # Compute value of nextState
        nextStateValue = self.computeValueFromQValues(nextState)

        # Batch update weights
        new_weights = util.Counter()

        for feature in self.features:
            new_weights[feature] = self.weights[feature] + self.alpha * ((reward + self.gamma * nextStateValue) - self.getQValue(state, action)) * self.features[feature]

        self.weights = new_weights

    def numStatesLearned(self):
        return None

    def save(self, i, j):

        # Build save file name
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD + '-iter-' + str(i + j)

        with open('save/' + fname + '.pickle', 'wb') as handle:
            pickle.dump(self.weights, handle)

    def load(self, fname):
        try:
            with open('save/' + fname, 'rb') as handle:
                self.Q = pickle.load(handle)
        except:
            ValueError('Failed to load file %s' % ('save/' + fname))
