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
            a_prime = self.computeActionFromQValues(s_prime)

            # If Mario is stuck, overwrite action with jump
            if self.s:
                mpos = feat.marioPosition(self.s.getTiles())

                if feat.stuck(self.s.getTiles(), mpos):
                    self.stuck_duration += 1

                    # If stuck for too long, rescue him
                    if self.stuck_duration > hp.STUCK_DURATION:
                        print "MODEL: Mario is stuck. Forcing jump to rescue..."

                        # On ground, get started with jump
                        if feat.groundVertDistance(self.s.getTiles(), mpos) == 0:
                            a_prime = random.choice([0, 10])
                        # Jump!
                        else:
                            a_prime = 10
                            self.jumps += 1

                        # Stop jumping and reset
                        if self.jumps > hp.MAX_JUMPS:
                            self.jumps = 0
                            self.stuck_duration = 0

            q_prime = self.getQ(s_prime, a_prime)

        # Only update if s exists (e.g., not first iteration of a_prime loop)
        if self.s:

            self.incN(self.s, self.a)

            q = self.getQ(self.s, self.a)

            # Add current features to trace
            features = feat.getFeatures(self.s, self.a)
            self.feature_traces.insert(0, features)

            # Batch update weights
            for i in xrange(len(self.feature_traces)):
                if (self.varlambda ** i) > hp.MIN_LAMBDA:
                    new_weights = util.Counter()
                    for ft in self.feature_traces[i]:
                        new_weights[ft] = self.weights[ft] + self.alpha * (self.varlambda ** i) * (self.r + self.gamma * q_prime - q) * features[ft]
                    self.weights = new_weights

        self.a = a_prime
        self.s = s_prime.copy()
        self.r = r_prime

        return self.a
