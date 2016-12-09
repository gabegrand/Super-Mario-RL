from approxQAgent import ApproxQAgent
import features as ft
import util

class ApproxSarsaAgent(ApproxQAgent):

    def getActionAndUpdate(self, state, reward):
        assert state
        assert isinstance(state, util.State)

        self.features = feat.getFeatures(state)
        action = self.computeActionFromQValues(state)

        # TODO can state ever be terminal?
        if self.prev_s:

            # Batch update weights
            new_weights = util.Counter()

            q_val = self.getQValue(state, action)
            for ft in self.features:
                new_weights[ft] = self.weights[ft] + self.alpha * self.getN(self.prev_s.getCurr(), self.prev_a) * (reward + self.gamma * q_val - self.weights[ft])
            self.weights = new_weights

        self.prev_a = action
        self.prev_s = state
        self.prev_r = reward

        self.incN(self.prev_s.getCurr(), self.prev_a)
        return self.prev_a