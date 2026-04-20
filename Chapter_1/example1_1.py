import numpy as np
import matplotlib.pyplot as plt

# Parameters
k1 = 3.0
k2 = 4.0

# Simulation settings
dt = 0.001
T = 10
N = int(T / dt)

def simulate(disturbance):
    x1 = 1.0
    x2 = -2.0
    t = 0.0

    t_vals = []
    x1_vals = []
    x2_vals = []

    for i in range(N):
        t_vals.append(t)
        x1_vals.append(x1)
        x2_vals.append(x2)

        # system dynamics
        f = disturbance(t)
        x1_dot = x2
        x2_dot = -k1*x1 - k2*x2 + f

        # Euler update
        x1 = x1 + dt * x1_dot
        x2 = x2 + dt * x2_dot

        t += dt

    return t_vals, x1_vals, x2_vals

# Case 1: no disturbance
t1, x1_1, x2_1 = simulate(lambda t: 0)

# Case 2: with disturbance
t2, x1_2, x2_2 = simulate(lambda t: np.sin(2*t))

# Plot
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t1, x1_1, label="x1")
plt.plot(t1, x2_1, label="x2")
plt.title("Without disturbance (f=0)")
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(t2, x1_2, label="x1")
plt.plot(t2, x2_2, label="x2")
plt.title("With disturbance (f = sin(2t))")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()