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

### Source file breakdown

- `abstractAgent.py` contains abstract class `AbstractAgent` for ensuring that all Q agents conform to the same methods.
- `approxQAgent.py` contains class `ApproxQAgent`, which implements Approximate Q Learning algorithm. Inherits from AbstractAgent.
- `approxSarsaAgent.py` contains class `ApproxSarsaAgent`, which implements Approximate SARSA algorithm. Inherits from ApproxQAgent.
- `feature_tests.py` is a scrapwork file for testing feature functionality for approximation algorithms.
- `features.py` contains the feature functions used for approximation algorithms.
- `heuristicAgent.py` contains class `HeuristicAgent`, which implements the basline Heuristic Agent described in our report.
- `hyperparameters.py` contains the hyperparameters set for testing, including number of iterations, level, agent type, and agent specific parameters.
- `qAgent.py` contains class `QLearningAgent`, which implements Exact Q Learning algorithm. Inherits from AbstractAgent.
- `randomAgent.py` contains the classes `RandomAgent` and child `WeightedRandomAgent`, which implement the basline random agents described in our report.
- `rewardModel.py` contains the functions that handle different scenarios for the reward function.
- `test.py` contains the main method from which testing is run.
- `util.py` contains the self-implemented state class, as well as the distribution utility code available via John DeNero and Dan Klein's UC Berkeley AI materials. For more information, see http://ai.berkeley.edu.

- `kill-mario.sh` is used within `test.py` to kill the FCEUX process.
- `mario.sh` can be used to start `test.py` as a background process.  It also kills `test.py` and the FCEUX application when they are active.