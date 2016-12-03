"""
Takes as input the default reward returned by the environment, and returns
a new reward based on our custom reward function.
"""

def rewardModel(reward, info, curr_score):

    # If Mario dies, punish
    if 'life' in info.keys() and info['life'] == 0:
        print("Oh no! Mario died!")
        reward -= hp.DEATH_PENALTY

    # If Mario's score increases, reward
    if 'score' in info.keys() and int(info['score']) > curr_score:
        delta = (int(info['score']) - curr_score) * hp.SCORE_FACTOR
        print("Bling! Score up by %d" % delta)
        reward += delta
        curr_score = int(info['score'])

    return reward
