from approxQAgent import ApproxQAgent
import features as feat
import util
import hyperparameters as hp

class ApproxSarsaAgent(ApproxQAgent):

    def getActionAndUpdate(self, state, reward):
        assert state
        assert isinstance(state, util.State)

        action = None

        # Terminal case
        if feat.marioPosition(state.getCurr()) is None:
            print('MODEL: Mario is dead. Returning action = None.')
            reward -= hp.DEATH_PENALTY

            # Batch update weights
            new_weights = util.Counter()
            for ft in self.features:
                new_weights[ft] = self.weights[ft] + self.alpha * reward * self.features[ft]
            self.weights = new_weights
        # Only update if prev_s exists (e.g., not first iteration of action loop)
        elif self.prev_s:
            action = self.computeActionFromQValues(state)

            prev_q = self.getQ(self.prev_s, self.prev_a)
            self.features = feat.getFeatures(state)

            # Batch update weights
            new_weights = util.Counter()

            q_val = self.getQ(state, action)
            for ft in self.features:
                new_weights[ft] = self.weights[ft] + self.alpha * (reward + self.gamma * q_val - prev_q) * self.features[ft]
            self.weights = new_weights
        #First iteration
        else:
            action = self.computeActionFromQValues(state)
            self.features = feat.getFeatures(state)

        self.prev_a = action
        self.prev_s = state
        self.prev_r = reward

        self.incN(self.prev_s.getCurr(), self.prev_a)
        return self.prev_a
