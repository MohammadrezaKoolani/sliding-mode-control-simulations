import numpy as np
import matplotlib.pyplot as plt

#============
# Parameters
#============
c = 1.5
c_bar = 10.0
beta = 30.0

x1_0 = 1.0
x2_0 = -2.0
u_0 = 0.0

dt = 1e-4
T = 10
N = int (T / dt)

#=============
# Disturbances
#==============
def disturbances(t):
    return np.sin(2.0 * t)

#===============
# Sign function 
#===============
def sign(s):
    if s > 0:
        return 1.0
    elif s < 0:
        return -1.0
    else:
        return 0.0
    
#==========
# Storage
#==========
t_vals = np.zeros(N)
x1_vals = np.zeros(N)
x2_vals = np.zeros(N)
u_vals = np.zeros(N)
v_vals = np.zeros(N)
sigma_vals = np.zeros(N)
s_vals = np.zeros(N)

#===================
# Initial Condition
#===================
x1 = x1_0
x2 = x2_0
u = u_0
t = 0.0

for k in range(N):
    f = disturbances(t)

    # Originakl sliding variable
    sigma = x2 + c * x1

    # Differentiate it
    sigma_dot = u + f + c * x2

    # Ausiliary sliding variable
    s = sigma_dot + c_bar * sigma

    # Control derivative
    v = -c * c_bar * x2 - (c + c_bar) * u - beta * sign(s)

    # save
    t_vals[k] = t
    x1_vals[k] = x1
    x2_vals[k] = x2
    u_vals[k] = u
    v_vals[k] = v
    sigma_vals[k] = sigma
    s_vals[k] = s

    # System dynamics
    x1_dot = x2
    x2_dot = u + f
    u_dot = v

    # Euler update
    x1 = x1 + dt * x1_dot
    x2 = x2 + dt * x2_dot
    u  = u  + dt * u_dot
    t  = t + dt


# =========================
# Fig. 1.15-like: control v
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, v_vals, linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('v')
plt.title('Example 1.4: Control v')

# =========================
# Fig. 1.16-like: physical control u = integral of v
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, u_vals, linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('u')
plt.title('Example 1.4: Physical control u = ∫v dt')

# =========================
# Fig. 1.17-like: sliding variables s and sigma
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, s_vals, label='s', linewidth=1.0)
plt.plot(t_vals, sigma_vals, label='sigma', linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Sliding variables')
plt.title('Example 1.4: Sliding variables')
plt.legend()

# =========================
# Fig. 1.18-like: state variables
# =========================
plt.figure(figsize=(8, 4))
plt.plot(t_vals, x1_vals, label='x1', linewidth=1.0)
plt.plot(t_vals, x2_vals, label='x2', linewidth=1.0)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('Example 1.4: State variables')
plt.legend()

plt.tight_layout()
plt.show()