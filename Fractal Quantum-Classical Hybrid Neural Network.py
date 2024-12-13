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

# Linear Qubit State Initialization (Real-valued representation)
states = np.random.normal(0, 1, size=(T, N))  # Linear amplitudes

# Normalize the linear states to ensure valid probabilities
def normalize(states):
    norm = np.linalg.norm(states, axis=1, keepdims=True)
    return states / norm

states = normalize(states)

# Function to compute entropy at each step
def compute_entropy(states):
    probabilities = states ** 2
    return -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)

# Compute entropy over time
entropy = compute_entropy(states)

# Build a Fractal Quantum-Classical Hybrid Neural Network
class FractalHybridNN(tf.keras.Model):
    def __init__(self, fractal_weights_scale=0.1):
        super(FractalHybridNN, self).__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.output_layer = Dense(1, activation='linear')
        self.fractal_weights_scale = fractal_weights_scale

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.output_layer(x)
        # Add fractal weight perturbation to simulate fractal space dynamics
        if training:
            fractal_noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.fractal_weights_scale)
            x += fractal_noise
        return x

# Prepare the dataset using linear qubit states
X_train = states
y_train = entropy[:, None]  # Use entropy as the target (reshaped for compatibility)

# Instantiate and compile the model
model = FractalHybridNN(fractal_weights_scale=0.05)
model.compile(optimizer='adam', loss='mse')

# Train the hybrid neural network
print("Training the Fractal Quantum-Classical Hybrid Neural Network...")
history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# Visualization Setup
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Correct initialization of the probability plot with proper data format
prob_plot = ax[0].imshow(np.abs(states), aspect='auto', cmap='plasma', origin='lower', extent=[0, N, 0, T])
ax[0].set_title('Linear Qubit State Distribution Over Fractal Space')
ax[0].set_xlabel('State Index')
ax[0].set_ylabel('Time')

entropy_line, = ax[1].plot([], [], lw=2, color='orange')
ax[1].set_xlim(0, T)
ax[1].set_ylim(0, np.max(entropy) * 1.1)
ax[1].set_title('Entropy Evolution as Fractal Dimension Stabilizes')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Entropy')

# Update function for the animation
def update(frame):
    prob_plot.set_data(np.abs(states[:frame, :]))  # Update data with correct absolute states
    entropy_line.set_data(range(frame), entropy[:frame])
    ax[1].fill_between(range(frame), 0, entropy[:frame], color='orange', alpha=0.3)
    return prob_plot, entropy_line

# Animate
ani = FuncAnimation(fig, update, frames=T, blit=True)
plt.tight_layout()
plt.show()
