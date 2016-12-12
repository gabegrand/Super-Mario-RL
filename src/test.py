"""
CS 182 Final Project
Super Mario Bros. Reinforcement Learning
Gabe Grand and Kevin Loughlin, 2016
"""

# OpenAI Gym Framework and Super Mario Bros
import gym
# Only need to pull this once
#import gym_pull
#gym_pull.pull('github.com/ppaquette/gym-super-mario@gabegrand')
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario import wrappers

# Python modules
import multiprocessing
import subprocess
import sys
import os
import numpy as np

# Our code
import hyperparameters as hp
import features as ft
from rewardModel import rewardModel
import util

# Agents
from qAgent import QLearningAgent
from approxQAgent import ApproxQAgent
from approxSarsaAgent import ApproxSarsaAgent
from randomAgent import RandomAgent
from heuristicAgent import HeuristicAgent

try:
    print('-- Creating environment')
    env = gym.make(hp.LEVEL)

    print('-- Acquiring multiprocessing lock')
    multiprocessing_lock = multiprocessing.Lock()
    env.configure(lock=multiprocessing_lock)

    # Discretize action space
    wrapper = wrappers.ToDiscrete()
    env = wrapper(env)

    print('-- Resetting environment')
    env.reset()

    # Initialize the correct agent
    agent = hp.AGENT_TYPE()
    print("-- Using %s" % agent.__class__.__name__)

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

    i = 1

    # Begin training loop
    while i <= hp.TRAINING_ITERATIONS:

        print('-- Resetting agent')
        agent.reset()
        rewardFunction.reset()

        print('-- START playing iteration %d / %d' % (i + j, hp.TRAINING_ITERATIONS + j))

        # Sample first action randomly
        action = env.action_space.sample()
        state = None
        while (ft.marioPosition(state) is None):
            state, reward, _, info = env.step(action)

        state = util.State(state, info['distance'], None)

        # Compute custom reward
        reward = rewardFunction.getReward(reward, info)

        dead = win = False

        # Begin main action-perception loop
        while not (info['iteration'] > i):

            # Check if Mario is at the end of the level
            if info['distance'] >= hp.LEVEL_WIN_DIST:
                win = True

            # Take NOOP action til environment ready to reset
            if dead or win:
                _, _, ready, _ = env.step(0)
                if ready:
                    break
            else:
                # Choose action according to Q
                action = agent.getActionAndUpdate(state, reward)

                # If Mario is off the screen, assume he is dead
                if action is None:
                    dead = True
                    continue

                # Take action
                nextState, reward, _, info = env.step(action)

                # Compute custom reward
                reward = rewardFunction.getReward(reward, info)

                # Advance the state
                state.step(nextState, info['distance'])

        # Handles ApproxSARSA too, since child of ApproxQ
        if isinstance(agent, ApproxQAgent):
            weights = agent.getWeights()
            formatted_weights = {}
            for w in weights:
                formatted_weights[w] = float("%.2f" % weights[w])
            print formatted_weights

        # Update diagnostics
        diagnostics[i] = {'states_learned': agent.numStatesLearned(),
                          'distance': info['distance'],
                          'score': info['score']}

        # print(info)
        print(diagnostics[i])

        # Save Q-values
        if i % hp.SAVE_EVERY == 0:
            print('Saving Q values...')
            agent.save(i, j, diagnostics)

        # Go to next iteration
        print('Iteration %d / %d complete.' % (i + j, hp.TRAINING_ITERATIONS + j))
        i = info['iteration'];

    print('-- Closing environment')
    env.close()

    print('-- DONE training iterations')
    print diagnostics

    # environment wasn't dying, so kill it
    subprocess.call(['./kill-mario.sh'])

# Die on interrupt
except KeyboardInterrupt:
    # environment wasn't dying, so kill it
    subprocess.call(['./kill-mario.sh'])
    print ""
    os._exit(0)
