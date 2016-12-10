from abstractAgent import *

class ApproxQAgent(AbstractAgent):

    def __init__(self):
        self.weights = util.Counter()
        self.N = util.Counter()
        self.alpha = hp.ALPHA
        self.gamma = hp.GAMMA
        self.actions = hp.MAPPING.keys()
        self.iter = 0

        self.r = None
        self.a = None
        self.s = None

    def getN(self, state, action):
        return self.N[str(state.getCurr()), action]

    def incN(self, state, action):
        self.N[str(state.getCurr()), action] += 1

    def getActionAndUpdate(self, s_prime, r_prime):
        assert s_prime
        assert isinstance(s_prime, util.State)

        action_should_be_none = False

        # Terminal case
        if feat.marioPosition(s_prime.getCurr()) is None:
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

            # Batch update weights
            new_weights = util.Counter()
            features = feat.getFeatures(self.s, self.a)
            for ft in features:
                new_weights[ft] = self.weights[ft] + self.alpha * (self.r + self.gamma * q_prime - q) * features[ft]
            self.weights = new_weights

        # UPDATE STATE, ACTION, REWARD

        # If Mario is dead
        if action_should_be_none:
            self.a = None
        # Otherwise, compute best action to take
        else:
            self.a = self.computeActionFromQValues(s_prime)

        # Store state and reward for next iteration
        self.s = s_prime
        self.r = r_prime

        return self.a

    def reset(self):
        self.s = None
        self.a = None
        self.r = None

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

        if util.flipCoin(max(hp.MIN_EPSILON, 50.0 / (50.0 + self.iter))):
            self.iter += 1
            return np.random.choice(self.actions, 1, p=hp.PRIOR)[0]
        """
          Compute the best action to take in a state.
        """

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

        return action

    def numStatesLearned(self):
        return None

    def getWeights(self):
        return self.weights

    def save(self, i, j, diagnostics):

        # Build save file name
        now = datetime.now()
        fname = '-'.join([str(x) for x in [now.year, now.month, now.day, now.hour, now.minute]]) + '-world-' + hp.WORLD + '-iter-' + str(i + j)

        saved_vals = {'weights': self.weights, 'N': self.N, 'diagnostics': diagnostics}

        with open('save/' + fname + '.pickle', 'wb') as handle:
            pickle.dump(saved_vals, handle)

    def load(self, fname):
        try:
            with open('save/' + fname, 'rb') as handle:
                saved_vals = pickle.load(handle)
                self.weights = saved_vals['weights']
                self.N = saved_vals['N']
        except:
            ValueError('Failed to load file %s' % ('save/' + fname))
