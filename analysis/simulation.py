from scipy.linalg import solve_discrete_lyapunov
import numpy as np
import pandas as pd


class VAR:
    def __init__(self, m, order=1):
        """
        Initialize a Vector Autoregressive VAR(p) model of form
        X_t = C_1 * X_t-1 + C_2 X_t-2 + ... + C_p * X_t-p + N_t

        Parameters:
        - m (int): The number of processes (variables).
        - order (int): The order of the VAR model (default is 1).

        Returns:
        - None
        """

        self.m = m
        self.order = order
        self.coefficients = []
        self.generated = False
        self.timeseries = []


    def generate(self, max_predictors):
        """
        Generate random coefficients for the VAR model while ensuring stability

        Parameters:
        - max_predictors (int): Maximum number of predictor variables for
        each variable in the model.

        Returns:
        - None
        """

        while not self.is_stable():
            self.coefficients = np.zeros((self.order, self.m, self.m))
            # assign coefficients randomly across all orders and variables
            for var in range(self.m):
                n_predictors = np.random.randint(0, max_predictors + 1)
                all_pairs = [(o, d) for o in range(self.order) for d in range(self.m)]
                np.random.shuffle(all_pairs)

                for (random_order, random_var) in all_pairs[:n_predictors]:
                    self.coefficients[random_order][var][random_var] = np.random.randn()

        self.generated = True


    def is_stable(self):
        """
        Checks whether the magnitude of the largest eigenvalue is less than 0.9,
        guaranteeing a stationary distribution.

        Returns:
        - boolean: True if model is stable, False otherwise.
        """
        companion_matrix = self.get_VAR1()
        eigenvalues = np.linalg.eigvals(companion_matrix)
        max = np.max(np.abs(eigenvalues))

        return True if max < 0.9 else False


    def simulate(self, n_steps, file_name=None):
        """
        Simulate data from the VAR model.

        Parameters:
        - n_steps (int): The number of time steps to simulate.
        - file_name (str): The name of the file to save the simulated data (optional).

        Returns:
        - timeseries (numpy.ndarray): Simulated time series data with shape (m, n_steps).
        """

        if not self.generated:
            self.generate(self.m / 2)

        # extra steps to ensure stationarity of timeseries
        n_stationary = 300
        timeseries = np.zeros((n_stationary + n_steps + + self.order, self.m))
        # first order values are random ~N(0,1)
        timeseries[:self.order, :] = np.random.randn(self.order, self.m)

        # timeseries when stationarity reached
        for t in range(self.order, n_steps + n_stationary + self.order):
            sum = np.zeros(self.m)
            for i in range(self.order):
                sum += np.dot(self.coefficients[i], timeseries[t - 1 - i, :])

            noise = np.random.randn(self.m)

            timeseries[t, :] = sum + noise

        # cut off time to stationarity
        self.timeseries = timeseries[n_stationary:]

        # safe to ./data/simulation if file_name given
        if file_name is not None:
            df = pd.DataFrame(timeseries)
            df.index = range(n_steps)
            df.to_csv(f'./data/simulations/{file_name}.csv')

        return self.timeseries[self.order:]


    def directed_information(self, x, y, z=[]):
        """
        Calculate I(X -> Y || Z), the Directed Information (DI) from variable X to Y
        causally conditioned on a set of variables Z.
        I(X -> Y || Z) = H(Y || Z) - H(Y || X, Z)
                       = log (std(N_t') / std(N_t))
        where H(Y || X, Z) = 0.5 * log (2 * pi * e * std(N_t)^2)
        Comparing two models, one given X and Z another one only given Z:
        Y_t = a * Y_t-1 + b * Z_t-1 + c * X_t-1 + ... + N_t
        Y_t = a' * Y_t-1 + b' * Z_t-1 + ... + N_t'

        Parameters:
        - x (int): Index of the source variable X.
        - y (int): Index of the target variable Y.
        - z (list): List of indices of variables in the set Z.

        Returns:
        - di (float): The calculated Directed Information.
        """

        steps = len(self.timeseries) - self.order
        # calculated with values of real simulation
        predicted_yxz = np.zeros((steps, 2 + len(z)))
        predicted_yz = np.zeros((steps, 1 + len(z)))

        yxz = [y] + [x] + z
        yz = [y] + z

        for t in range(steps):
            sum_yxz = np.zeros(2 + len(z))
            sum_yz = np.zeros(1 + len(z))

            for i in range(self.order):
                sum_yxz += np.dot(self.coefficients[i][np.ix_(yxz, yxz)], self.timeseries[self.order - 1 + t - i, yxz])
                sum_yz += np.dot(self.coefficients[i][np.ix_(yz, yz)], self.timeseries[self.order - 1 + t - i, yz])

            predicted_yxz[t, :] = sum_yxz
            predicted_yz[t, :] = sum_yz

        residuals_y_given_xz = self.timeseries[self.order:, y] - predicted_yxz[:, 0]
        residuals_y_given_z = self.timeseries[self.order:, y] - predicted_yz[:, 0]

        std_residuals_given_xz = np.std(residuals_y_given_xz)
        std_residuals_given_z = np.std(residuals_y_given_z)

        di = np.log(std_residuals_given_z / std_residuals_given_xz)

        return di

    def get_VAR1(self):
        """
        Convert the VAR(p) model to VAR(1) in companion form.

        Returns:
        - F (numpy.ndarray): The coefficient matrix for VAR(1) in companion form.
        """

        if self.order == 1:
            return self.coefficients[0]

        F_upper = np.hstack(self.coefficients)
        F_lower = np.eye(self.m * (self.order - 1), self.m * self.order)
        F = np.vstack([F_upper, F_lower])
        return F


    def __repr__(self):
        rep = f'VAR({self.order}): '
        rep += 'X_t = '
        for i in range(self.order):
            rep += f'C_{i + 1} * X_t-{i + 1} + '
        rep += 'N_t\n'

        for i in range(self.order):
            rep += f'C_{1 + i} = \n' + self.coefficients[i].__repr__() + '\n'

        return rep
