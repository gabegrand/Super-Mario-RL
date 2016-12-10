from approxQAgent import *

class ApproxSarsaAgent(ApproxQAgent):

    def getActionAndUpdate(self, s_prime, r_prime):
        assert s_prime
        assert isinstance(s_prime, util.State)

        a_prime = None

        # Terminal case
        if feat.marioPosition(s_prime.getTiles()) is None:
            print('MODEL: Mario is dead. Returning a_prime = None.')
            r_prime -= hp.DEATH_PENALTY
            q_prime = r_prime
        else:
            a_prime = self.computea_primeFromQValues(s_prime)
            q_prime = self.getQ(s_prime, a_prime)

        # Only update if s exists (e.g., not first iteration of a_prime loop)
        if self.s:

            self.incN(self.s, self.a)

            q = self.getQ(self.s, self.a)

            # Batch update weights
            new_weights = util.Counter()
            features = feat.getFeatures(self.s, self.a)
            for ft in features:
                new_weights[ft] = self.weights[ft] + self.alpha * (self.r + self.gamma * q_prime - q) * features[ft]
            self.weights = new_weights

        self.a = a_prime
        self.s = s_prime.copy()
        self.r = r_prime

        return self.a
