# Super-Mario-RL
## CS 182 Artificial Intelligence Final Project
## By Gabe Grand and Kevin Loughlin

### Setup instructions for Mac
Our instructions assume that you have Python 2.7 and Homebrew installed on your machine. In order to use our system, you must download and install the OpenAI Gym, the FCEUX Nintendo Entertainment System emulator, and the Gym Super Mario environment. Then, open a bash shell, run the following commands.

    pip install gym
    pip install gym-pull
    brew upgrade
    brew install homebrew/games/fceux

Once this is completed, open a python shell and run the following.

    import gym
    import gym_pull
    gym_pull.pull('github.com/ppaquette/gym-super-mario@gabegrand')

Assuming everything installs correctly, navigate to `src/`. You can set the hyperparameters as you wish in `hyperparameters.py`, and then run `test.py` (which will launch the emulator and use our code for training).  Finally, you can kill the processes via Control-C in the Python terminal, or running `./mario.sh` in the src directory.