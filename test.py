# SEE README.md FOR PREREQUISITE INSTALL INSTRUCTIONS

import gym
import gym_pull
from ppaquette_gym_super_mario import wrappers
import multiprocessing
from qAgent import QLearningAgent
from qAgent import ApproxQAgent
from qAgent import ApproxSarsaAgent
import hyperparameters as hp
import features as ft
import numpy as np

# Initialize the correct agent
agent = None
if hp.AGENT_TYPE == 0:
    agent = QLearningAgent()
    print "USING EXACT Q AGENT"
elif hp.AGENT_TYPE == 1:
    agent = ApproxQAgent()
    print "USING APPROX Q AGENT"
elif hp.AGENT_TYPE == 2:
    agent = ApproxSarsaAgent()
    print "USING APPROX SARSA AGENT"
else:
    raise ValueError("Invalid AGENT_TYPE in hyperparameters")

# Load from previous saved Q values
if hp.LOAD_FROM is not None:

    print('Loading Q values from %s' % hp.LOAD_FROM)
    agent.load(hp.LOAD_FROM)

    # Start iterations from where we left off
    j = int(hp.LOAD_FROM[hp.LOAD_FROM.rfind('-')+1:hp.LOAD_FROM.rfind('.pickle')])
    print('Starting at iteration %d' % j)

else:
    j = 0

# Diagnostics
diagnostics = {}
num_freezes = 0

print('-- START training iterations')
i = 1
while i <= hp.TRAINING_ITERATIONS:

    print('-- START creating environment')
    env = gym.make(hp.LEVEL)
    print('-- DONE creating environment')

    print('-- START acquiring multiprocessing lock')
    multiprocessing_lock = multiprocessing.Lock()
    env.configure(lock=multiprocessing_lock)
    print('-- DONE acquiring multiprocessing lock')

    # Discretize action space to 14 possible button combinations
    wrapper = wrappers.ToDiscrete()
    env = wrapper(env)

    print('-- START resetting environment')
    env.reset()
    print('-- DONE resetting environment')

    print('-- START playing iteration %d / %d' % (i + j, hp.TRAINING_ITERATIONS + j))
    done = False

    # Keep track of agent's score in game
    curr_score = 0

    # Sample first action randomly
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    # Choose action according to Q
    action = agent.getAction(state)

    while not done:
        # Take action
        nextState, reward, done, info = env.step(action)

        # Only factored into update for Sarsa
        nextAction = None
        if hp.AGENT_TYPE == 2:
            nextAction = agent.getAction(nextState)

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

        # Update Q values; nextAction only used in Sarsa
        agent.update({'state': state, 'action': action, 'nextState': nextState, 'nextAction': nextAction, 'reward': reward})

        # Advance the state and action
        state = nextState

        # if sarsa, update differently
        if hp.AGENT_TYPE == 2:
            action = nextAction
        else:
            action = agent.getAction(nextState)
        

    # Handle case where game gets stuck
    if 'ignore' in info.keys() and info['ignore'] == True:
        print('Game stuck. Resetting...')
        num_freezes += 1
    else:
        # Update diagnostics
        diagnostics[i] = {'freezes': num_freezes,
                          'states_learned': agent.numStatesLearned(),
                          'distance': info['distance'],
                          'score': info['score'],
                          }

        print(info)
        print(diagnostics[i])

        # Save Q-values
        if i % hp.SAVE_EVERY == 0:
            print('Saving Q values...')
            agent.save(i, j)

        # Go to next iteration
        print('Iteration %d / %d complete.' % (i + j, hp.TRAINING_ITERATIONS + j))
        i += 1

    print('-- DONE playing')
    env.close()

print('-- DONE training iterations')
print diagnostics
