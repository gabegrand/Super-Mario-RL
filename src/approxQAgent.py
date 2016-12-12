from abstractAgent import *

# Implements Approximate Q Agent
class ApproxQAgent(AbstractAgent):

    def __init__(self):
        self.weights = util.Counter()
        self.N = util.Counter()
        self.alpha = hp.ALPHA
        self.gamma = hp.GAMMA
        self.varlambda = hp.LAMBDA
        self.actions = hp.MAPPING.keys()
        self.iter = 0
        self.feature_traces = []

        self.stuck_duration = 0
        self.jumps = 0

        self.r = None
        self.a = None
        self.s = None

    def getN(self, state, action):
        return self.N[str(state.getTiles()), action]

    def incN(self, state, action):
        self.N[str(state.getTiles()), action] += 1

    def getActionAndUpdate(self, s_prime, r_prime):
        assert s_prime
        assert isinstance(s_prime, util.State)

        action_should_be_none = False
        curr_features = None

        # Terminal case
        mpos = feat.marioPosition(s_prime.getTiles())
        if mpos is None:
            print('MODEL: Mario is dead. Returning action = None.')
            action_should_be_none = True
            r_prime -= hp.DEATH_PENALTY

        # Only update if s exists (e.g., not first iteration of action loop)
        if self.s:
            self.incN(self.s, self.a)

            q = self.getQ(self.s, self.a)

            # Get Q value of s_prime a_prime
            if action_should_be_none:
                q_prime = r_prime
            else:
                q_prime = self.computeValueFromQValues(s_prime)

            # Add current features to trace
            curr_features = feat.getFeatures(self.s, self.a)
            self.feature_traces.insert(0, curr_features)

            # Maintain max length of trace list
            if len(self.feature_traces) > hp.MAX_TRACES:
                del self.feature_traces[-1]

            # Batch update weights for each set of features in eligibility trace
            for i, features in enumerate(self.feature_traces):
                new_weights = util.Counter()
                for ft in features:
                    new_weights[ft] = self.weights[ft] + self.alpha * (self.varlambda ** i) * (self.r + self.gamma * q_prime - q) * features[ft]
                self.weights = new_weights

        # UPDATE STATE, ACTION, REWARD

        # If Mario is dead
        if action_should_be_none:
            self.a = None

        # If situation normal, compute best action to take
        else:
            self.a = self.computeActionFromQValues(s_prime)

            # If Mario is stuck, overwrite action with jump
            if curr_features and bool(curr_features['stuck']):
                self.stuck_duration += 1

                # If stuck for too long, rescue him
                if self.stuck_duration > hp.STUCK_DURATION:
                    print "MODEL: Mario is stuck. Forcing jump to rescue..."

                    # On ground, get started with jump
                    if feat.groundVertDistance(self.s.getTiles(), mpos) == 0:
                        self.a = random.choice([0, 10])
                    # Jump!
                    else:
                        self.a = 10
                        self.jumps += 1

                    # Stop jumping and reset
                    if self.jumps > hp.MAX_JUMPS:
                        self.jumps = 0
                        self.stuck_duration = 0

        # Store state and reward for next iteration
        self.s = s_prime.copy()
        self.r = r_prime

        return self.a

    def reset(self):
        self.s = None
        self.a = None
        self.r = None
        self.stuck_duration = 0
        self.jumps = 0
        self.feature_traces = []

    def getQ(self, state, action):
        return self.weights * feat.getFeatures(state, action)

    def computeValueFromQValues(self, state):

        # Keep track of values of each action
        action_values = util.Counter()

        # Get value of each action
        for action in self.actions:
            # avoid dividing by zero by adding 1
            action_values[action] = self.getQ(state, action)

        # Return max value
        return action_values[action_values.argMax()]

    def computeActionFromQValues(self, state):

        if util.flipCoin(max(hp.MIN_EPSILON, hp.EP_DEC / (hp.EP_DEC + self.iter))):
            self.iter += 1
            a = np.random.choice(self.actions, 1, p=hp.PRIOR)[0]
            # print(str(a) + '*')
            return a

        # Keep track of values of each action
        action_values = []

        # Get value of each action
        for action in self.actions:
            action_values.append(self.getQ(state, action) + hp.K / (self.getN(state, action) + 1.0))

        # Compute max value over all actions
        max_value = max(action_values)

        # Get indices of all actions that lead to max value
        indices = [i for i, x in enumerate(action_values) if x == max_value]

        # Return action with max value, breaking ties randomly
        action = self.actions[random.choice(indices)]

        # print action

        return action

    def numStatesLearned(self):
        return len(self.N)

    def getWeights(self):
        return self.weights

    def save(self, i, j, diagnostics):

        # Build save file name
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD_STR + '-iter-' + str(i + j)

        saved_vals = {'weights': self.weights, 'iter': self.iter, 'diagnostics': diagnostics}

        with open('save/' + fname + '.pickle', 'wb') as handle:
            pickle.dump(saved_vals, handle)

        with open('save/' + fname + '-N' + '.pickle', 'wb') as handle:
            pickle.dump(self.N, handle)

    def load(self, fname):
        try:
            with open('save/' + fname, 'rb') as handle:
                saved_vals = pickle.load(handle)
                self.weights = saved_vals['weights']
                self.iter = saved_vals['iter']
        except:
            ValueError('Failed to load file %s' % ('save/' + fname))

        try:
            with open('save/' + fname + '-N', 'rb') as handle:
                self.N = pickle.load(handle)
        except:
            ValueError('Failed to load file %s' % ('save/' + fname + '-N'))
