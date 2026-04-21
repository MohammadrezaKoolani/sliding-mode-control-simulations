import numpy as np
import matplotlib.pyplot as plt

# =========================
# Example 1.5 parameters
# =========================
c = 1.5
alpha = 2.0
tau = 0.01

x1_0 = 1.0
x2_0 = -2.0

dt = 1e-4
T = 5.0
N = int(T / dt)

def disturbance(t):
    return np.sin(2.0 * t)

def sign(s):
    if s > 0:
        return 1.0
    elif s < 0:
        return -1.0
    else:
        return 0.0

# storage
t_vals = np.zeros(N)
x1_vals = np.zeros(N)
x2_vals = np.zeros(N)
sigma_vals = np.zeros(N)
u_vals = np.zeros(N)

z_vals = np.zeros(N)
ueq_hat_vals = np.zeros(N)
ueq_ideal_vals = np.zeros(N)
f_hat_vals = np.zeros(N)
f_true_vals = np.zeros(N)

# initial conditions
x1 = x1_0
x2 = x2_0
z = 0.0
t = 0.0

for k in range(N):
    f = disturbance(t)
    sigma = x2 + c * x1

    # actual sliding mode control
    u = -c * x2 - alpha * sign(sigma)

    # low-pass filter: tau * z_dot = -z + sign(sigma)
    z_dot = (-z + sign(sigma)) / tau

    # equivalent control estimate
    ueq_hat = -c * x2 - alpha * z

    # ideal equivalent control from the known disturbance
    ueq_ideal = -c * x2 - f

    # disturbance estimate
    f_hat = -alpha * z

    # save
    t_vals[k] = t
    x1_vals[k] = x1
    x2_vals[k] = x2
    sigma_vals[k] = sigma
    u_vals[k] = u

    z_vals[k] = z
    ueq_hat_vals[k] = ueq_hat
    ueq_ideal_vals[k] = ueq_ideal
    f_hat_vals[k] = f_hat
    f_true_vals[k] = f

    # system dynamics
    x1_dot = x2
    x2_dot = u + f

    # Euler update
    x1 = x1 + dt * x1_dot
    x2 = x2 + dt * x2_dot
    z  = z  + dt * z_dot
    t  = t  + dt

# =========================
# Plot 1: Equivalent control estimation
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, ueq_ideal_vals, label='ideal equivalent control', linewidth=1.0)
plt.plot(t_vals, ueq_hat_vals, '--', label='estimated equivalent control', linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel(r'$u_{eq}$')
plt.title('Example 1.5: Equivalent control estimation')
plt.legend()
plt.tight_layout()

# =========================
# Plot 2: Disturbance estimation
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, f_true_vals, label='true disturbance', linewidth=1.0)
plt.plot(t_vals, f_hat_vals, '--', label='estimated disturbance', linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('disturbance')
plt.title('Example 1.5: Disturbance estimation')
plt.legend()
plt.tight_layout()

plt.show()