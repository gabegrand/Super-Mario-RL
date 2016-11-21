import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario import wrappers
import multiprocessing
from agents import QLearningAgent
from hyperparameters import *

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

# Initialize agent
Agent = QLearningAgent()

print('-- START training iterations')
for i in xrange(TRAINING_ITERATIONS):

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

        print newState, reward

        # Update Q values
        Agent.update(state, action, newState, reward)

        # Advance the state
        state = newState

    print(info)

    # Handle case where game gets stuck
    if 'ignore' in info.keys() and info['ignore'] == True:
        i -= 1
        print "Game stuck. Resetting..."

    print('-- DONE playing')

print('-- DONE training iterations')
