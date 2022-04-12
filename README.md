# Deep Q Networks

This is a project that implements deep q-networks using pycharm.


## Package Installation Process
The easiest way to make sure this package is properly installed is to create a new virtual environment for this installation. Then follow the steps below.
1. Clone this repository into a folder <*folder-directory*>
2. Open a terminal in that folder and run:
```pip install -e .```
3. Install the proper version of pytorch for your virtual environment, using the command provided in https://pytorch.org/. For refference, the command used to setup the virtual environment during development was ```pip3 install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html```, but different hardware and environments (virtualenv vs conda) require a different personalized command.
4. Download the Roms from atari from https://github.com/openai/atari-py#roms and follow the instructions for **installation from source**.
5. Follow the installation instructions (in the **Installation** section) from https://github.com/openai/baselines to install the atari baselines.

## Installing from testpypi:
```pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dqn==x.x.x```

## Notes on the versions of this project:
 - 0.0.1 - Initial version of the project
 - 0.1.0 - Added Q-learning
 - 0.1.1 - Fixed bug in last release (the file with the implemented QLearningAgent had not been added to the commit.)
 - 0.1.2 - Fixed a small issue with the q-learning algorithm and added a feature that allows it to count iterations as episodes or timesteps.
 - 0.1.3 - Another small fix.
 - 0.1.4 - Added an epsilon greedy policy with non-linear decay.
 - 0.1.5 - Added yet another epsilon greedy policy, where the decay is done by steps: every few timesteps, the epsilon decays a value.
 - 0.1.6 - Fixed issues with Linear DQNs and added a test example.
 - 0.2.0 - Added DoubleDQNAgent. Renamed the AtariDQNPolicy to a more general and appropriate name: TrainEvalPolicy.
 - 0.2.1 - Small modifications to make the agents more adaptable to other environments besides gym ones. 
 - 0.3.0 - LinearDQN is now implemented as GeneralLinearDQN was. Updates were made in the DQNAgent and DoubleDQNAgent.
 - 0.4.0 - CustomLinearDQN added. In a later update it will completely replace LinearDQN. DoubleDQNAgent readded. Apparently it had been removed by accident in the previous version.
 - 0.4.1 - Bugfix in DQNAgent.
 - 0.4.2 - Changed the name if the class AtariDNQEnv to the intended name AtariDQNEnv
 - 0.4.3 - Added evaluation episodes, like in the original deep mind article. The best performing agent is saved.