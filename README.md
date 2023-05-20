An implementation of the PPO algorithm written in Python using Pytorch. 
The actor and critic networks are a simple MLP with one hidden layer of size 64. The environment is fully observable; i.e. obs = [cos(angle), sin(angle), angular velocity].

https://github.com/Ezgii/PPO-on-pendulum/assets/4748948/357aa43d-b6ad-4810-b855-b0a725aaed5a

### Loss functions and Learning curve:

![figure1](https://github.com/Ezgii/PPO-on-pendulum/blob/main/results/figure1_True_True.png)

### Value grid:

![figure2](https://github.com/Ezgii/PPO-on-pendulum/blob/main/results/figure2_True_True.png)
