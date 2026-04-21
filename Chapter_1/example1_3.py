import numpy as np
import matplotlib.pyplot as plt

#============
# Parameters
#============
c = 1.5
alpha = 2.0
epsilon = 0.01

x1_0 = 1.0
x2_0 = -2.0

dt = 1e-4
T = 10.0
N = int(T / dt)

##=============
# Disturbances
#==============
def disturbance(t):
    return np.sin(2.0 * t)

    
#=========
# Storage
#=========
t_vals = np.zeros(N)
x1_vals = np.zeros(N)
x2_vals = np.zeros(N)
sigma_vals = np.zeros(N)
u_vals = np.zeros(N)

#===================
# Initial Condition
#===================
x1 = x1_0
x2 = x2_0
t = 0.0

#============
# Simulation
#============
for k in range(N):
    sigma = x2 + c * x1
    smooth_sign = sigma / (abs(sigma) + epsilon)
    
    u = -c * x2 - alpha * smooth_sign
    f = disturbance(t)

    # save the values
    t_vals[k] = t
    x1_vals[k] = x1
    x2_vals[k] = x2
    sigma_vals[k] = sigma
    u_vals[k] = u

    # System dynamics
    x1_dot = x2
    x2_dot = u + f
    
    # Euler update
    x1 = x1 + dt * x1_dot
    x2 = x2 + dt * x2_dot
    t = t + dt

# =========================
# Plot 1: smooth control
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, u_vals, linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('u')
plt.title('Example 1.3: Smooth control')
plt.tight_layout()

# =========================
# Plot 2: sliding variable
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, sigma_vals, linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel(r'$\sigma$')
plt.title('Example 1.3: Sliding variable')
plt.tight_layout()

# =========================
# Plot 3: state variables
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, x1_vals, label=r'$x_1$', linewidth=1.0)
plt.plot(t_vals, x2_vals, label=r'$x_2$', linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('Example 1.3: Time history of the state variables')
plt.legend()
plt.tight_layout()

# =========================
# Plot 4: phase portrait
# =========================
plt.figure(figsize=(5, 5))
plt.plot(x1_vals, x2_vals, linewidth=1.0)
plt.grid(True)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Example 1.3: Phase portrait')

x1_line = np.linspace(min(x1_vals), max(x1_vals), 400)
x2_line = -c * x1_line
plt.plot(x1_line, x2_line, '--', linewidth=1.0, label=r'$\sigma=0$')
plt.legend()
plt.tight_layout()

plt.show()