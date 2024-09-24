import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt


sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

h = 0.001 
hsamp = 0.1
transient_time = 500.0  


def lorenz_system(state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def runge_kutta_step(state, h, sigma, rho, beta):
    k1 = lorenz_system(state, sigma, rho, beta)
    k2 = lorenz_system(state + 0.5 * h * k1, sigma, rho, beta)
    k3 = lorenz_system(state + 0.5 * h * k2, sigma, rho, beta)
    k4 = lorenz_system(state + h * k3, sigma, rho, beta)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


np.random.seed(10)
initial_condition = np.random.rand(3) * 20 - 10
state = initial_condition.copy()


num_transient_steps = int(transient_time / h)
for _ in range(num_transient_steps):
    state = runge_kutta_step(state, h, sigma, rho, beta)


sampling_steps = int(hsamp / h)
trajectory = []
num_sampling_steps = 20000

for _ in range(num_sampling_steps):
    for _ in range(sampling_steps):
        state = runge_kutta_step(state, h, sigma, rho, beta)
    trajectory.append(state)

trajectory = np.array(trajectory)
x_component = trajectory[:, 0]


min_val = np.min(x_component)
max_val = np.max(x_component)
scaled_x_component = 2 * (x_component - min_val) / (max_val - min_val) - 1


num_masks = 10
T = 10000
N_v = 30
lengthened_scaled_x_component = np.repeat(scaled_x_component, T)

masks = [np.random.uniform(-1, 1, N_v) for _ in range(num_masks)]
masked_components = np.zeros((num_masks, len(lengthened_scaled_x_component)))

for m_idx in range(num_masks):
    for i in range(0, len(lengthened_scaled_x_component), T):
        for n in range(N_v):
            start_idx = i + n * (T // N_v)
            end_idx = i + (n + 1) * (T // N_v)
            masked_components[m_idx, start_idx:end_idx] = lengthened_scaled_x_component[start_idx:end_idx] * masks[m_idx][n]

masked_lengthened_x_component_median = np.median(masked_components, axis=0)
input_data = masked_lengthened_x_component_median


@njit
def hermite_interpolation(p0, m0, p1, m1, dt, t=0.5):
    return (2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 * dt + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * m1 * dt

@njit
def derivatives(y, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, interpol=False, dt=0.01, t=0.5):
    A, phi = y
    A_delay, phi_delay = delay_data

    if interpol:
        A_delayed = hermite_interpolation(A_delay[0], A_delay[1], A_delay[2], A_delay[3], dt, t)
        phi_delayed = hermite_interpolation(phi_delay[0], phi_delay[1], phi_delay[2], phi_delay[3], dt, t)
    else:
        A_delayed = A_delay[0]
        phi_delayed = phi_delay[0]

    dA_dt = pSL * A + gamma_real * A**3 + kappa * A_delayed * np.cos(phi_fb + phi_delayed - phi)
    dphi_dt = 1.0 + gamma_imag * A**2 + (kappa * A_delayed / A) * np.sin(phi_fb + phi_delayed - phi)

    return np.array([dA_dt, dphi_dt])

@njit
def rk4_step(y, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, dt):
    k1 = dt * derivatives(y, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, False, dt)
    y_mid = y + 0.5 * k1
    k2 = dt * derivatives(y_mid, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, True, dt, t=0.5)
    y_mid = y + 0.5 * k2
    k3 = dt * derivatives(y_mid, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, True, dt, t=0.5)
    k4 = dt * derivatives(y + k3, delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, False, dt)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@njit
def solve(input_data, pSL_base, eta, dt, tau, gamma_real, gamma_imag, kappa, phi_fb):
    time = np.arange(0, len(input_data) * dt, dt)
    delay_steps = int(tau / dt)

    states = np.zeros((len(time), 2), dtype=np.float64)  
    states[0] = [np.abs(np.random.rand() + 1j * np.random.rand()), np.angle(np.random.rand() + 1j * np.random.rand())]

    for i in range(1, len(time)):
        delay_index = max(0, i - delay_steps)
        delay_data = (
            np.array([states[delay_index, 0], (states[delay_index, 0] - states[max(0, delay_index - 1), 0]) / dt,
                      states[max(0, delay_index - 1), 0], (states[max(0, delay_index - 1), 0] - states[max(0, delay_index - 2), 0]) / dt]),
            np.array([states[delay_index, 1], (states[delay_index, 1] - states[max(0, delay_index - 1), 1]) / dt,
                      states[max(0, delay_index - 1), 1], (states[max(0, delay_index - 1), 1] - states[max(0, delay_index - 2), 1]) / dt])
        )
        pSL = pSL_base + eta * input_data[i]
        states[i] = rk4_step(states[i - 1], delay_data, pSL, gamma_real, gamma_imag, kappa, phi_fb, dt)

    A = states[:, 0]
    phi = states[:, 1]
    X = A * np.exp(1j * phi)
    return X, time

tau = 100
pSL_base = 0.2
gamma_real = -0.1
gamma_imag = 0.0
kappa = 0.1
phi_fb = 0
dt = 0.01
T = 10000
eta = 1


X, time = solve(input_data, pSL_base, eta, dt, tau, gamma_real, gamma_imag, kappa, phi_fb)


plt.figure(figsize=(10, 6))
plt.plot(time[:T], np.real(X[:T]), label='Real part')
plt.plot(time[:T], np.imag(X[:T]), label='Imaginary part')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Time Series of Real and Imaginary Parts of Stuart-Landau Oscillator with $\\tau = 100$')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(np.real(X[:T]), np.imag(X[:T]), label='Phase space')
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('Phase Space of Stuart-Landau Oscillator with $\\tau = 100$')
plt.grid(True)
plt.show()
