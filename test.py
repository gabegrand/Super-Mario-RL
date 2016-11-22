import gym
import gym_pull
# Only need to pull this once
# gym_pull.pull('github.com/ppaquette/gym-super-mario')
from ppaquette_gym_super_mario import wrappers
import multiprocessing
from agents import QLearningAgent
from hyperparameters import *

# Initialize agent
Agent = QLearningAgent()

# Diagnostics
diagnostics = {'states_learned': 0,
               'freezes': 0,
               'alpha': ALPHA,
               'epsilon': EPSILON,
               'gamma': GAMMA
               }

print('-- START training iterations')
i = 0
while i < TRAINING_ITERATIONS:

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

    # Sample first action randomly
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    state = str(state)

    while not done:

        # Choose action according to Q
        action = Agent.getAction(state)

        # Take action
        newState, reward, done, info = env.step(action)

        # Convert numpy arrays to strings for storage in dict
        state = str(state)
        newState = str(newState)

        # Update Q values
        Agent.update(state, action, newState, reward)

        # Advance the state
        state = newState

    # Handle case where game gets stuck
    if 'ignore' in info.keys() and info['ignore'] == True:
        print('Game stuck. Resetting...')
        diagnostics['freezes'] += 1

    # Save Q-values and go to next iteration
    else:
        print('Iteration %d complete. Saving Q values...' % i)
        Agent.save(i)
        i += 1

    # Print diagnostic information
    diagnostics['states_learned'] = Agent.numStatesLearned()
    print(info)
    print (diagnostics)

    print('-- DONE playing')
    env.close()

print('-- DONE training iterations')
