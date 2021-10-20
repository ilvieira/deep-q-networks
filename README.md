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
