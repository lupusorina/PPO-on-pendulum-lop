### Project description
An implementation of the PPO algorithm written in Python using Pytorch. 
The actor and critic networks are a simple MLP with one hidden layer of size 64. The environment is fully observable; i.e. obs = [cos(angle), sin(angle), angular velocity].


https://github.com/Ezgii/PPO-on-pendulum/assets/4748948/41de8653-96e0-4fb9-8197-9ed033b5214e


### Environment
[OpenAI's Gym](https://gym.openai.com/) is a framework for training reinforcement 
learning agents. It provides a set of environments and a
standardized interface for interacting with those.   
In this project, I used the Pendulum environment from gym.

### Installation

#### Using conda (recommended)    
1. [Install Anaconda](https://www.anaconda.com/products/individual)

2. Create the env    
`conda create a1 python=3.8` 

3. Activate the env     
`conda activate a1`    

4. install torch ([steps from pytorch installation guide](https://pytorch.org/)):    
- if you don't have an nvidia gpu or don't want to bother with cuda installation:    
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`    
  
- if you have an nvidia gpu and want to use it:    
[install cuda](https://docs.nvidia.com/cuda/index.html)   
install torch with cuda:   
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

5. other dependencies   
`conda install -c conda-forge matplotlib gym opencv pyglet`

#### Using pip
`python3 -m pip install -r requirements.txt`

### How to run the code
On terminal, write:

`python3 main.py`

### Results

#### Loss functions and Learning curve:

![figure1](https://github.com/Ezgii/PPO-on-pendulum/blob/main/results/figure1_True_True.png)

#### Value grid:

![figure2](https://github.com/Ezgii/PPO-on-pendulum/blob/main/results/figure2_True_True.png)
