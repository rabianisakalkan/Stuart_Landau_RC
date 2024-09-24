import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

class PolarComplexDE:
    def __init__(self, pSL, omega, gamma, kappa, phi_fb, tau, dt=0.01):
        self.pSL = pSL
        self.omega = omega
        self.gamma = gamma
        self.kappa = kappa
        self.phi_fb = phi_fb
        self.tau = tau
        self.dt = dt

    def hermite_interpolation(self, p0, m0, p1, m1, t=0.5):
        return (2 * t**3 - 3 * t**2 + 1) * p0 + (t**3 - 2 * t**2 + t) * m0 * self.dt + (-2 * t**3 + 3 * t**2) * p1 + (t**3 - t**2) * m1 * self.dt

    def derivatives(self, y, delay_data, interpol=False, t=0.5):
        A, phi = y
        A_delay, phi_delay = delay_data

        if interpol:
            A_delayed = self.hermite_interpolation(A_delay[0], A_delay[1], A_delay[2], A_delay[3], t)
            phi_delayed = self.hermite_interpolation(phi_delay[0], phi_delay[1], phi_delay[2], phi_delay[3], t)
        else:
            A_delayed = A_delay[0]
            phi_delayed = phi_delay[0]

        dA_dt = self.pSL * A + self.gamma.real * A**3 + self.kappa * A_delayed * np.cos(self.phi_fb + phi_delayed - phi)
        dphi_dt = self.omega + self.gamma.imag * A**2 + (self.kappa * A_delayed / A) * np.sin(self.phi_fb + phi_delayed - phi)

        return np.array([dA_dt, dphi_dt])

    def rk4_step(self, y, delay_data, dt):
        k1 = dt * self.derivatives(y, delay_data)
        y_mid = y + 0.5 * k1
        k2 = dt * self.derivatives(y_mid, delay_data, interpol=True, t=0.5)
        y_mid = y + 0.5 * k2
        k3 = dt * self.derivatives(y_mid, delay_data, interpol=True, t=0.5)
        k4 = dt * self.derivatives(y + k3, delay_data)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self):
        time = np.arange(0, 20000, self.dt)
        delay_steps = int(self.tau / self.dt)

        states = np.zeros((len(time), 2), dtype=float)
        states[0] = [np.abs(np.random.rand() + 1j * np.random.rand()), np.angle(np.random.rand() + 1j * np.random.rand())]

        derivatives = np.zeros_like(states)

        for i in range(1, len(time)):
            delay_index = max(0, i - delay_steps)
            delay_data = (
                [states[delay_index, 0], (states[delay_index, 0] - states[max(0, delay_index - 1), 0]) / self.dt,
                 states[max(0, delay_index - 1), 0], (states[max(0, delay_index - 1), 0] - states[max(0, delay_index - 2), 0]) / self.dt],
                [states[delay_index, 1], (states[delay_index, 1] - states[max(0, delay_index - 1), 1]) / self.dt,
                 states[max(0, delay_index - 1), 1], (states[max(0, delay_index - 1), 1] - states[max(0, delay_index - 2), 1]) / self.dt]
            )

            states[i] = self.rk4_step(states[i - 1], delay_data, self.dt)
            derivatives[i] = self.derivatives(states[i], delay_data)

        A = states[:, 0]
        phi = states[:, 1]
        X = A * np.exp(1j * phi)
        X_abs = A**2
        dX_dt = derivatives[:, 0] * np.exp(1j * phi) + 1j * A * derivatives[:, 1] * np.exp(1j * phi)
        return X, X_abs, dX_dt, time

    @staticmethod
    def count_unique_maxima(x, tolerance=1e-2):
        unique_maxima = np.array([])
        if np.all(np.isclose(x, x[0], atol=tolerance)):
            num_maxima = 0
        else:
            max_indices = np.where((np.roll(x, 1) < x) & (np.roll(x, -1) < x))[0]
            max_indices = max_indices[(max_indices != 0) & (max_indices != len(x)-1)]
            unique_maxima = np.unique(np.round(x[max_indices], int(np.ceil(-np.log10(tolerance)))))
        num_maxima = len(unique_maxima)
        return unique_maxima, num_maxima

    @staticmethod
    def count_unique_minima(x, tolerance=1e-2):
        unique_minima = np.array([])
        if np.all(np.isclose(x, x[0], atol=tolerance)):
            num_minima = 0
        else:
            min_indices = np.where((np.roll(x, 1) > x) & (np.roll(x, -1) > x))[0]
            unique_minima = np.unique(np.round(x[min_indices], int(np.ceil(-np.log10(tolerance)))))
        num_minima = len(unique_minima)
        return unique_minima, num_minima

def simulate_tau_intensity(params):
    tau, phi_fb, pSL = params
    solver = PolarComplexDE(pSL=pSL, omega=1.0, gamma=-0.1, kappa=0.1, phi_fb=phi_fb, tau=tau)
    X, X_abs, dX_dt, time = solver.solve()


    intensity = X_abs
    intensity_transient = intensity[round(17000 / solver.dt):]  

    unique_maxima, num_maxima = PolarComplexDE.count_unique_maxima(intensity_transient)
    unique_minima, num_minima = PolarComplexDE.count_unique_minima(intensity_transient)

    if num_maxima > 0:
        max_value = max(unique_maxima)
    else:
        max_value = intensity[round(17000 / solver.dt)]

    if num_minima > 0:
        min_value = min(unique_minima)
    else:
        min_value = intensity[round(17000 / solver.dt)]

    return tau, max_value, min_value, intensity, dX_dt

def plot_bifurcation_diagram_tau_intensity():
    mesh = 50
    tolerance = 1e-2
    phi_fb = 0.0
    pSL = 0.05  
    tau_values = np.linspace(5, 50, mesh)  
    params = [(tau, phi_fb, pSL) for tau in tau_values]

    with Pool() as p:
        results = p.map(simulate_tau_intensity, params)

    taus, maxima, minima, intensity_all, dX_dt_all = zip(*results)

    plt.figure(figsize=(10, 6))
    plt.plot(taus, maxima, '-', linewidth=2, label='Maxima')
    plt.plot(taus, minima, '-', linewidth=2, label='Minima')
    plt.title(f'Bifurcation Diagram for Intensity $|X|^2$ as a Function of $\\tau$')
    plt.xlabel(f'$\\tau$', fontsize=16)
    plt.ylabel(f'Intensity $|X|^2$', fontsize=16)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_bifurcation_diagram_tau_intensity()
