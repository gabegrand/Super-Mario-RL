import util
import hyperparameters as hp
import numpy as np
import pickle
from datetime import datetime
import random
import features as feat

# ensures that all Q agents conform to the same methods
class AbstractAgent:

    def getActionAndUpdate(self, state, reward):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def getN(self, state, action):
        raise NotImplementedError()

    def incN(self, state, action):
        raise NotImplementedError()

    def getQ(self, state, action):
        raise NotImplementedError()

    def computeValueFromQValues(self, state):
        raise NotImplementedError()

    def computeActionFromQValues(self, state):
        raise NotImplementedError()

    def save(self, i, j, diagnostics):
        raise NotImplementedError()

    def load(self, fname):
        raise NotImplementedError()