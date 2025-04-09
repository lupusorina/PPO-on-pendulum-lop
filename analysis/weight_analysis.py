import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
FOLDER = "../results/"

# Pick the latest folder.
folders = os.listdir(FOLDER)
folders = [f for f in folders if os.path.isdir(os.path.join(FOLDER, f))]
latest_folder = max(folders, key=lambda x: os.path.getmtime(os.path.join(FOLDER, x)))
FOLDER = os.path.join(FOLDER, latest_folder, "policy_net")
print(FOLDER)

# Plot the weights of the policy net.

policies = os.listdir(FOLDER)
policies = [p for p in policies if p.endswith(".pt")]

weights = []
biases = []
for policy in tqdm(policies, desc="Loading policies"):
    policy_path = os.path.join(FOLDER, policy)
    policy_net = torch.load(policy_path)
    weights_layer_1 = policy_net["layer1.weight"].detach().numpy()
    print(weights_layer_1.flatten())
    bias_layer_1 = policy_net["layer1.bias"].detach().numpy()
    weights.append(weights_layer_1.flatten())
    biases.append(bias_layer_1.flatten())

fig, axs = plt.subplots(2, 1)
axs[0].plot(weights)
axs[1].plot(biases)
plt.show()

