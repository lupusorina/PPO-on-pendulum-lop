import pandas as pd

one_hidden_layer_csv = pd.read_csv('bad_fit_1.csv')
two_hidden_layer_csv = pd.read_csv('bad_fit_2.csv')
three_hidden_layer_csv = pd.read_csv('bad_fit_3.csv')
four_hidden_layer_csv = pd.read_csv('bad_fit_4.csv')

# Plot the data
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))
plt.plot(one_hidden_layer_csv['hidden_size'], one_hidden_layer_csv['bad_fit'], 'o-', label='1 hidden layer')
plt.plot(two_hidden_layer_csv['hidden_size'],  two_hidden_layer_csv['bad_fit'],  'o-',label='2 hidden layers')
plt.plot(three_hidden_layer_csv['hidden_size'], three_hidden_layer_csv['bad_fit'],  'o-', label='3 hidden layers')
plt.plot(four_hidden_layer_csv['hidden_size'], four_hidden_layer_csv['bad_fit'], 'o-', label='4 hidden layers')
plt.xlabel('Hidden layer size')
plt.ylabel('Number of bad fits')
plt.title('Number of bad fits vs Nb. of weights in the hidden layer')
plt.legend()
plt.grid(True)
plt.savefig('bad_fit.png')
plt.show()
