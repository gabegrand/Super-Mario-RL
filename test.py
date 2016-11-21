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

    print('-- START playing')
    done = False

    # Sample first action randomly
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    state = str(state)

    while not done:

        # Choose action according to Q
        action = Agent.getAction(state)
        # action = hyperparameters.mapping[action] TODO

        newState, reward, done, info = env.step(action)

        state = str(state)
        newState = str(newState)

        print newState, reward

        # Update Q values
        Agent.update(state, action, newState, reward)

        state = newState

    print(info)

    # TODO: Handle case where info[ignore] is true

    print('-- DONE playing')

print('-- DONE training iterations')
