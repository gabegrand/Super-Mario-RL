import hyperparameters as hp

"""
Takes as input the default reward returned by the environment, and returns
a new reward based on our custom reward function.
"""

class rewardModel:

    def __init__(self):
        self.curr_score = 0
        self.crossedGap = False

    def getReward(self, reward, info):

        # Encourage Mario to go right
        # if reward > 0:
        #     reward *= 10
        # # Penalize Mario for standing still
        # else:
        #     reward -= 1
        #     reward *= 10

        if reward <= 0:
            reward -= 1

        # Get Mario across checkpoint gap in World 1-1
        if hp.WORLD == '1-1' and not self.crossedGap and info['distance'] > 1425:
            print "Crossed gap! Reward +500!"
            reward += 500
            self.crossedGap = True

        """ This functionality has been moved to getActionAndUpdate"""
        # If Mario dies, punish
        # if 'life' in info.keys() and info['life'] == 0:
        #     print("Oh no! Mario died!")
        #     reward -= hp.DEATH_PENALTY

        # If Mario's score increases, reward
        if 'score' in info.keys() and int(info['score']) > self.curr_score:
            delta = (int(info['score']) - self.curr_score) * hp.SCORE_FACTOR
            print("Bling! Score up by %d" % delta)
            reward += delta
            self.curr_score = int(info['score'])

        return reward

    def reset(self):
        self.crossedGap = False
