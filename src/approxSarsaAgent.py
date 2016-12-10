from approxQAgent import *

class ApproxSarsaAgent(ApproxQAgent):

    def getActionAndUpdate(self, s_prime, r_prime):
        assert s_prime
        assert isinstance(s_prime, util.s_prime)

        action = None

        # Terminal case
        if feat.marioPosition(s_prime.getCurr()) is None:
            print('MODEL: Mario is dead. Returning action = None.')
            r_prime -= hp.DEATH_PENALTY
            q_prime = r_prime
        else:
            action = self.computeActionFromQValues(s_prime)
            q_prime = self.getQ(s_prime, action)

        # Only update if prev_s exists (e.g., not first iteration of action loop)
        if self.prev_s:

            self.incN(state, action)

            q = self.getQ(self.prev_s, self.prev_a)

            # Batch update weights
            new_weights = util.Counter()
            features = feat.getFeatures(self.s, self.a)
            for ft in self.prev_feat:
                new_weights[ft] = self.weights[ft] + self.alpha * (self.r + self.gamma * q_prime - q) * features[ft]
            self.weights = new_weights

        self.a = action
        self.s = s_prime
        self.r = r_prime

        return self.a
