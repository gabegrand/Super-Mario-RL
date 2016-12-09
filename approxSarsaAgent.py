from approxQAgent import ApproxQAgent
import features as feat
import util

class ApproxSarsaAgent(ApproxQAgent):

    def getActionAndUpdate(self, state, reward):
        assert state
        assert isinstance(state, util.State)

        if feat.marioPosition(state.getCurr()) is None:
            action = None
        else:
            action = self.computeActionFromQValues(state)

        # TODO can state ever be terminal?
        if self.prev_s:
            prev_q = self.getQ(self.prev_s, self.prev_a)
            if action is not None:
                self.features = feat.getFeatures(state)

            # Batch update weights
            new_weights = util.Counter()

            q_val = self.getQ(state, action)
            for ft in self.features:
                new_weights[ft] = self.weights[ft] + self.alpha * self.getN(self.prev_s.getCurr(), self.prev_a) * (reward + self.gamma * q_val - prev_q) * self.features[ft]
            self.weights = new_weights
        elif action is not None:
            self.features = feat.getFeatures(state)

        self.prev_a = action
        self.prev_s = state
        self.prev_r = reward

        self.incN(self.prev_s.getCurr(), self.prev_a)
        return self.prev_a