import util
import random
import numpy as np
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

    def save(self, i):
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD + '-iter-' + str(i)
        np.save('save/' + fname, self.Q)
