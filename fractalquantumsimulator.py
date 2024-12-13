import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Parameters for the simulation
N = 1000  # Number of informational nodes (classical equivalent of qubits)
T = 200   # Number of time steps
D_critical = 2.4  # Critical fractal dimension for stabilization

# Initialize the fractal dimension and linear qubit amplitudes
D_f = np.linspace(1.0, 3.0, T)  # Fractal dimension evolving over time
entropy_gradients = np.random.normal(0, 1, size=(T, N))  # Random perturbations

# Fractal Quantum Simulator: Initialize Quantum States
states_real = np.random.normal(0, 1, size=(T, N))  # Real part of quantum states
states_imag = np.random.normal(0, 1, size=(T, N))  # Imaginary part of quantum states
states = states_real + 1j * states_imag

# Normalize the quantum states to ensure valid probabilities
def normalize(states):
    norm = np.linalg.norm(states, axis=1, keepdims=True)
    return states / norm

states = normalize(states)

# Function to compute entropy at each step
def compute_entropy(states):
    probabilities = np.abs(states) ** 2
    return -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)

# Function to apply fractal perturbations over time
def fractal_perturbation(states, D_f, scale=0.05):
    perturbation = np.random.normal(0, scale, size=states.shape) * (D_f[:, None] - D_critical)
    return states + perturbation

# Compute entropy over time and perturb states dynamically
entropy = np.zeros(T)
for t in range(T):
    states[t] = normalize(fractal_perturbation(states[t], D_f[t:t+1]))
    entropy[t] = compute_entropy(states[t:t+1])[0]

# Visualization Setup
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Quantum State Probability Distribution
prob_plot = ax[0].imshow(np.abs(states), aspect='auto', cmap='plasma', origin='lower', extent=[0, N, 0, T])
ax[0].set_title('Fractal Quantum State Distribution Over Time')
ax[0].set_xlabel('State Index')
ax[0].set_ylabel('Time')

# Entropy Evolution Visualization
entropy_line, = ax[1].plot([], [], lw=2, color='orange')
ax[1].set_xlim(0, T)
ax[1].set_ylim(0, np.max(entropy) * 1.1)
ax[1].set_title('Entropy Evolution as Fractal Perturbations Stabilize Quantum States')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Entropy')

# Update function for the animation
def update(frame):
    prob_plot.set_data(np.abs(states[:frame, :]))  # Update probability data
    entropy_line.set_data(range(frame), entropy[:frame])
    ax[1].fill_between(range(frame), 0, entropy[:frame], color='orange', alpha=0.3)
    return prob_plot, entropy_line

# Animate
ani = FuncAnimation(fig, update, frames=T, blit=True)
plt.tight_layout()
plt.show()
