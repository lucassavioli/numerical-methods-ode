class RungeKutta4thOrder:
    def __init__(self, f):
        self.f = f

    def solve(self, y0, t0, tn, h):
        """
        Solves the differential equation using the 4th-order Runge-Kutta method.

        Parameters:
        - y0: Initial value of the dependent variable
        - t0: Initial value of the independent variable
        - tn: Final value of the independent variable
        - h: Step size

        Returns:
        - t_values: List of time values
        - y_values: List of corresponding solution values
        """
        t_values = [t0]
        y_values = [y0]

        while t_values[-1] < tn:
            t_n = t_values[-1]
            y_n = y_values[-1]

            k1 = h * self.f(t_n, y_n)
            k2 = h * self.f(t_n + 0.5 * h, y_n + 0.5 * k1)
            k3 = h * self.f(t_n + 0.5 * h, y_n + 0.5 * k2)
            k4 = h * self.f(t_n + h, y_n + k3)

            y_n1 = y_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t_values.append(t_n + h)
            y_values.append(y_n1)

        return t_values, y_values

# Example of usage:

# Define the ODE: dy/dt = t - y
def ode_function(t, y):
    return t - y

# Create an instance of the RungeKutta4thOrder class with the ODE function
rk = RungeKutta4thOrder(ode_function)

# Set initial conditions and solve the ODE
initial_value = 1.0
initial_time = 0.0
final_time = 2.0
step_size = 0.2

t_values, y_values = rk.solve(initial_value, initial_time, final_time, step_size)

# Print the results
for t, y in zip(t_values, y_values):
    print(f"t = {t:.2f}, y = {y:.4f}")
