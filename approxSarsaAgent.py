from approxQAgent import ApproxQAgent
import features as ft
import util

class ApproxSarsaAgent(ApproxQAgent):

    # only method that is different
    # update_dict must consist of state, action, nextState, nextAction, reward
    def update(self, update_dict):

        state = update_dict['state']
        action = update_dict['action']
        nextState = update_dict['nextState']
        nextAction = update_dict['nextAction']
        reward = update_dict['reward']

        self.N[str(state), action] += 1

        # Ensure Mario is on the screen in both states
        # if ft.marioPosition(state) and ft.marioPosition(nextState):
        if ft.marioPosition(nextState):

            # Update features
            features = ft.getFeatures(nextState, action)
            self.features = features
        else:
            print "update: Mario not on screen"

        # Update prev state
        self.prev_state = state

        # Compute value of nextState SARSA style
        nextStateValue = self.getQValue(state, nextAction)

        # Batch update weights
        new_weights = util.Counter()

        for feature in self.features:
            new_weights[feature] = self.weights[feature] + self.alpha * ((reward + self.gamma * nextStateValue) - self.getQValue(state, action)) * self.features[feature]

        self.weights = new_weights
        print self.weights
