# Reinforcement Learning with a Custom Neural Network Library

I developed this neural network library that includes common activation functions and optimisers. I have only implemented fully-connected layers as the main goal of the project was for implementing reinforcement learning algorithms for control purposes using the OpenAI gym. A write-up of some results that I obtained and a more detailed explanation of the algorithms and neural network library can be found [here](<./Learning to Walk - A Reinforcement Learning Approach.pdf>).

# Neural Network Library
An example of using the neural network library for regression is given in `nn_bivariate_regression.py`.

# Reinforcement Learning Algorithms
The following reinforcement learning algorithms have been implemented.

* TD3
* A2C
* DDPG
* GAE
* REINFORCE (with and without baseline)

# Dependencies
* `Python 3`
* `Matplotlib`
* `Numpy`
* `OpenAI gym`
* `Imageio` (gif output)

# Running an example

```sh
# Working config for Mountain Car Continuous
./main.py DDPG MountainCarContinuous-v0 --lr 1e-3 --lr_c 1e-3 --num_episodes=1000 --hidden_layers 32

# Working config for Lunar Lander
python3 main.py GAE LunarLander-v2 --lr=1e-4 --lr_c=1e-4 --num_episodes=5000 --batch_size=16 --hidden_layers 20 --gamma=0.99 --Lambda=0.95

# Working config for Pendulum-v0
python3 main.py DDPG Pendulum-v0 --lr=1e-3 --lr_c=1e-2 --num_episodes=10000 --batch_size=64 --hidden_layers 20 20 --gamma=0.99

# Working config for BipedalWalker-v3
python3 main.py TD3 BipedalWalker-v3 --lr=1e-5 --lr_c=1e-4 --num_episodes=5000 --batch_size=64 --hidden_layers 200 200 --gamma=0.997 --replay_mem_size=6e5
```

# Bipedal Walker Environment
Below is a Gif of a TD3 agent solving the [bipedal walker environment](<https://github.com/openai/gym/wiki/BipedalWalker-v2>) with a score of 319.


![TD3 Bipedal Walker Gif](<./TD3-319.09.gif>)