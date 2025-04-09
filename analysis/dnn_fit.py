import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime
import pandas as pd

RESULTS = 'results'
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)
time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
if not os.path.exists(os.path.join(RESULTS, time_now)):
    os.makedirs(os.path.join(RESULTS, time_now))
FOLDER_RESULTS = os.path.join(RESULTS, time_now)
ABS_FOLDER_RESUlTS = os.path.abspath(FOLDER_RESULTS)
FOLDER_RESTORE_CHECKPOINT = os.path.abspath(RESULTS + '/20250318-173452/000151388160')
print(f"Saving results to {ABS_FOLDER_RESUlTS}")

# Generate random data.
def generate_data(n_samples=1000):
    x = np.linspace(-5, 5, n_samples)
    y = 2 * np.sin(x) + 0.5 * np.exp(-x**2 / 10) + 0.1 * np.random.randn(n_samples)
    # Linear function.
    # y = 2 * x + 0.1 * np.random.randn(n_samples)
    return x, y

# Define the neural network
class SimpleDNN(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[1, 1], output_size=1):
        super(SimpleDNN, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# Main function
def train(seed=1, hidden_sizes=[1, 1]):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Generate data
    x, y = generate_data()

    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)

    # Create model.
    model = SimpleDNN(input_size=1, hidden_sizes=hidden_sizes, output_size=1)
    # Init weights and biases to small random values.
    model.model[0].weight.data = torch.randn(model.model[0].weight.data.size()) * 0.01
    model.model[0].bias.data = torch.randn(model.model[0].bias.data.size()) * 0.01

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    n_epochs = 1000
    weights = []
    biases = []
    for epoch in range(n_epochs):
        weights.append(model.model[0].weight.detach().numpy().flatten())
        biases.append(model.model[0].bias.detach().numpy().flatten())

        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # Print progress every 100 epochs
        # if (epoch + 1) % 100 == 0:
        #     print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model.
    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor).numpy()
        
    return x, y, y_pred, weights, biases


if __name__ == "__main__":
    MAX_SEED = 20
    MAX_HIDDEN_SIZE = 20
    NB_HIDDEN_LAYERS = 4

    list_of_seeds = np.arange(1, MAX_SEED)
    hidden_sizes = np.arange(1, MAX_HIDDEN_SIZE)

    counter_bad_fit_list = []

    for hidden_size in hidden_sizes:
        window_max_list = []
        y_pred_list = []
        plt.figure(figsize=(10, 6))
        x, y = generate_data()
        plt.scatter(x, y, alpha=0.1, label='Data points')
        hidden_size_list = [hidden_size for _ in range(NB_HIDDEN_LAYERS)]
        print(hidden_size_list)

        for seed in list_of_seeds:
            x, _, y_pred, weights, bias = train(seed, hidden_sizes=hidden_size_list)  
            # check how many of them have a long window of close to 0 values.
            window_max = 0
            for i, y in enumerate(y_pred):
                if np.abs(y_pred[i] - y_pred[i-1]) < 0.001:
                    window += 1
                if np.abs(y_pred[i] - y_pred[i-1]) > 0.001:
                    window = 0
                if window > window_max:
                    window_max = window  
            window_max_list.append(window_max)
            y_pred_list.append(y_pred)
                        
            plt.plot(x, y_pred, '-', label='DNN fit_seed=' + str(seed))

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('DNN Fitting to Random Data for Hidden size ' + str(hidden_size))
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(ABS_FOLDER_RESUlTS + '/dnn_fit_' + str(hidden_size) + '.png')
        
        # Plot the ones whose window list is more than 500.
        print(f"Window max list: {window_max_list}")
        counter_bad_fit = 0
        plt.figure(figsize=(10, 6))
        for i, window_max in enumerate(window_max_list):
            if window_max > 450:
                plt.plot(x, y_pred_list[i], '-', label='DNN fit_seed=' + str(i))
                counter_bad_fit += 1
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('DNN Fitting to Random Data for Hidden size ' + str(hidden_size))
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.savefig(ABS_FOLDER_RESUlTS + '/dnn_fit_window_' + str(hidden_size) + '.png')
        
        print(f"Number of bad fits: {counter_bad_fit} for hidden size {hidden_size}")
        counter_bad_fit_list.append(counter_bad_fit)
        
    df = pd.DataFrame({'hidden_size': hidden_sizes, 'bad_fit': counter_bad_fit_list})
    df.to_csv(ABS_FOLDER_RESUlTS + '/bad_fit_' + str(NB_HIDDEN_LAYERS) + '.csv', index=False)
        
    # plot counter bad fit
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_sizes, counter_bad_fit_list, '-o')
    plt.xlabel('Hidden size')
    plt.ylabel('Number of bad fits')
    plt.title('Number of bad fits vs Hidden size')
    plt.grid(True)
    plt.savefig(ABS_FOLDER_RESUlTS + '/bad_fit_nb_hidden_layers_' + str(NB_HIDDEN_LAYERS) + '.png')

