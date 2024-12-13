import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the simulation
N = 1000  # Number of quantum states (informational nodes)
T = 200   # Number of time steps
D_critical = 2.4  # Critical fractal dimension where stabilization occurs

# Initialize the fractal dimension and quantum state amplitudes
D_f = np.linspace(1.0, 3.0, T)  # Fractal dimension evolving over time
entropy_gradients = np.random.normal(0, 1, size=(T, N))  # Random perturbations for entropy gradients

# Quantum State Initialization
states_real = np.random.normal(0, 1, size=(T, N))  # Real part of quantum amplitudes
states_imag = np.random.normal(0, 1, size=(T, N))  # Imaginary part of quantum amplitudes
quantum_states = states_real + 1j * states_imag

# Normalize the quantum states to ensure valid probabilities
def normalize(states):
    norm = np.linalg.norm(states, axis=1, keepdims=True)
    return states / norm

quantum_states = normalize(quantum_states)

# Function to compute entropy at each step
def compute_entropy(states):
    probabilities = np.abs(states) ** 2
    return -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)

# Compute entropy over time
entropy = compute_entropy(quantum_states)

# Visualization Setup
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
prob_plot = ax[0].imshow(np.abs(quantum_states) ** 2, aspect='auto', cmap='plasma', extent=[0, N, 0, T])
ax[0].set_title('Quantum State Probability Distribution Over Fractal Space')
ax[0].set_xlabel('State Index')
ax[0].set_ylabel('Time')

entropy_line, = ax[1].plot([], [], lw=2, color='orange')
ax[1].set_xlim(0, T)
ax[1].set_ylim(0, np.max(entropy))
ax[1].set_title('Entropy Evolution as Fractal Dimension Stabilizes')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Entropy')

# Update function for the animation
def update(frame):
    prob_plot.set_data(np.abs(quantum_states[frame:frame+1]) ** 2)
    entropy_line.set_data(range(frame), entropy[:frame])
    ax[1].fill_between(range(frame), 0, entropy[:frame], color='orange', alpha=0.3)
    return prob_plot, entropy_line

# Animate
ani = FuncAnimation(fig, update, frames=T, blit=True)
plt.tight_layout()
plt.show()
