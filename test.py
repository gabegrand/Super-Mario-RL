# SEE README.md FOR PREREQUISITE INSTALL INSTRUCTIONS

import gym
import gym_pull
from ppaquette_gym_super_mario import wrappers
import multiprocessing
from qAgent import QLearningAgent
from approxQAgent import ApproxQAgent
from approxSarsaAgent import ApproxSarsaAgent
import hyperparameters as hp
import features as ft
from rewardModel import rewardModel
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

# Initialize reward function
rewardFunction = rewardModel()

# Diagnostics
diagnostics = {}
num_freezes = 0

print('-- START training iterations')
i = 1
while i <= hp.TRAINING_ITERATIONS:

    # Initialize environment
    print('-- Creating environment...')
    env = gym.make(hp.LEVEL)

    print('-- Acquiring multiprocessing lock')
    multiprocessing_lock = multiprocessing.Lock()
    env.configure(lock=multiprocessing_lock)

    # Discretize action space to 14 possible button combinations
    wrapper = wrappers.ToDiscrete()
    env = wrapper(env)

    print('-- Resetting environment')
    env.reset()

    print('-- Resetting agent')
    agent.reset()

    print('-- START playing iteration %d / %d' % (i + j, hp.TRAINING_ITERATIONS + j))
    done = dead = False

    # Sample first action randomly
    action = env.action_space.sample()
    state, reward, _, info = env.step(action)

    # Choose action according to Q
    action = agent.getAction(state)

    # Repeat each action for set number of timesteps
    action_counter = hp.ACTION_DURATION

    while not done:

        # Take action
        nextState, reward, dead, info = env.step(action)
        action_counter -= 1

        # Update Q values and compute next action
        if action_counter <= 0:

            # Check if Mario is dead
            if dead:
                done = True

            # Compute custom reward
            reward = rewardFunction.getReward(reward, info)

            # Only factored into update for Sarsa
            nextAction = None
            if hp.AGENT_TYPE == 2:
                nextAction = agent.getAction(nextState)

            # Update Q values; nextAction only used in Sarsa
            agent.update({'state': state,
                          'action': action,
                          'nextState': nextState,
                          'nextAction': nextAction,
                          'reward': reward})

            # Advance the state and action
            state = nextState

            # Choose next action according to Q. If SARSA, nextAction has already been chosen.
            if hp.AGENT_TYPE == 2:
                action = nextAction
            else:
                action = agent.getAction(nextState)
            action_counter = hp.ACTION_DURATION

    # Handle case where game gets stuck
    if 'ignore' in info.keys() and info['ignore']:
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
            agent.save(i, j, diagnostics[i])

        print agent.getWeights()

        # Go to next iteration
        print('Iteration %d / %d complete.' % (i + j, hp.TRAINING_ITERATIONS + j))
        i += 1

    print('-- Closing environment')
    env.close()

print('-- DONE training iterations')

print diagnostics
