# SEE README.md FOR PREREQUISITE INSTALL INSTRUCTIONS

import gym
import gym_pull
from ppaquette_gym_super_mario import wrappers
import multiprocessing
from agents import QLearningAgent
from hyperparameters import *

# Initialize q learning agent
agent = QLearningAgent()

# Diagnostics
diagnostics = {}
num_freezes = 0

print('-- START training iterations')
i = 1
while i <= TRAINING_ITERATIONS:

    print('-- START creating environment')
    env = gym.make(LEVEL)
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

    print('-- START playing iteration %d' %i)
    done = False

    # Keep track of agent's score in game
    curr_score = 0

    # Sample first action randomly
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    state = str(state)

    while not done:

        # Choose action according to Q
        action = agent.getAction(state)

        # Take action
        newState, reward, done, info = env.step(action)

        # Convert numpy arrays to strings for storage in dict
        state = str(state)
        newState = str(newState)

        # If Mario dies, punish
        if 'life' in info.keys() and info['life'] == 0:
            print("Oh no! Mario died!")
            reward -= DEATH_PENALTY

        # If Mario's score increases, reward
        if 'score' in info.keys() and int(info['score']) > curr_score:
            delta = (int(info['score']) - curr_score) * SCORE_FACTOR
            print("Bling! Score up by %d" % delta)
            reward += delta
            curr_score = int(info['score'])

        # Update Q values
        agent.update(state, action, newState, reward)

        # Advance the state
        state = newState

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

        # Save Q-values and go to next iteration
        print('Iteration %d complete. Saving Q values...' % i)
        agent.save(i)
        i += 1

    print('-- DONE playing')
    env.close()

print('-- DONE training iterations')
print diagnostics
