# Super-Mario-RL
# CS 182 Artificial Intelligence Final Project
# By Gabe Grand and Kevin Loughlin

# Setup instructions for Mac
First, install the OpenAI Gym and the FCEUX Nintendo Entertainment System emulator.

    pip install gym
    pip install gym-pull
    brew upgrade
    brew install homebrew/games/fceux

Next, open a Python environment in Terminal and execute the following commands.

    import gym
    import gym_pull
    gym_pull.pull('github.com/ppaquette/gym-super-mario@gabegrand')

Assuming everything installs correctly, you should be good to go! Navigate to the `Super-Mario-RL` directory and run `python test.py`.
