import matplotlib.pyplot as plt
import numpy as np

def plot_function(a_limit):
    # 1. Create the range of values from 0 to 'a'
    # we use 400 points to ensure a smooth curve
    value = np.linspace(-a_limit, a_limit, 400)
    
    # 2. Define the equation (e.g., b = np.exp(value))
    b = -np.exp(abs(value))/60
    c=  np.exp(-2*abs(value))-1
    # 3. Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(value, b, label="Smooth Reward", color='blue', linewidth=2)
    plt.plot(value, c, label="Angle Reward", color='red', linewidth=2)
    
    # 4. Add labels and title
    plt.title(f'Plot of reward_functions from -{a_limit} to {a_limit}')
    plt.xlabel('Value')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 5. Save the result
    plt.savefig('exponential_plot.png')
    # plt.show() # Uncomment if running locally to see the window

# Example use: Plotting from 0 to 5
plot_function(6)