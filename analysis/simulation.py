import numpy as np
import pandas as pd
import analysis.utils as utils


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
        self.di_matrix = np.zeros((m, m))


    def generate(self, max_predictors):
        """
        Generate random coefficients for the VAR model while ensuring stability

        Parameters:
        - max_predictors (int): Maximum number of predictor variables for
        each variable in the model.

        Returns:
        - None
        """

        while True:
            self.coefficients = np.zeros((self.order, self.m, self.m))
            # assign coefficients randomly across all orders and variables
            for var in range(self.m):
                n_predictors = np.random.randint(0, max_predictors + 1)
                all_pairs = [(o, d) for o in range(self.order) for d in range(self.m)]
                np.random.shuffle(all_pairs)

                for (random_order, random_var) in all_pairs[:n_predictors]:
                    self.coefficients[random_order][var][random_var] = np.random.randn()

            # rescale VAR if possible
            if self.order == 1:
                self.rescale()
                break

            if self.is_stable():
                break

        self.generated = True


    def is_stable(self):
        """
        Checks whether the magnitude of the largest eigenvalue is less or equal to 0.9,
        guaranteeing a stationary distribution.

        Returns:
        - boolean: True if model is stable, False otherwise.
        """
        companion_matrix = self.get_VAR1()
        eigenvalues = np.linalg.eigvals(companion_matrix)
        max = np.max(np.abs(eigenvalues))

        return True if max <= 0.9 else False


    def rescale(self):
        """
        Rescales the coefficient matrix for the largest eigenvalue to be 0.9.
        This is only possible if the largest eigenvalue is not zero.

        Returns:
        - None
        """
        companion_matrix = self.get_VAR1()
        eigenvalues = np.linalg.eigvals(companion_matrix)
        max = np.max(np.abs(eigenvalues))

        # value = factor * max
        factor = (0.9 / max) if max != 0 else 1
        self.coefficients[0] *= factor


    def simulate(self, n_steps, file_name=None):
        """
        Simulate time series from the VAR model. Noise ~N(0, 1/4)

        Parameters:
        - n_steps (int): The number of time steps to simulate.
        - file_name (str): The name of the file to save the simulated data (optional).

        Returns:
        - timeseries (numpy.ndarray): Simulated time series data with shape (m, n_steps).
        """

        if not self.generated:
            self.generate(self.m * self.order / 2)

        # extra steps to ensure stationarity of timeseries
        n_stationary = 300
        timeseries = np.zeros((n_stationary + n_steps + self.order, self.m))
        # first order values are random ~N(0,1)
        timeseries[:self.order, :] = np.random.randn(self.order, self.m)

        for t in range(self.order, n_steps + n_stationary + self.order):
            sum = np.zeros(self.m)
            for i in range(self.order):
                sum += self.coefficients[i] @ timeseries[t - 1 - i, :]

            noise = np.random.normal(scale=0.25, size=self.m)

            timeseries[t, :] = sum + noise

        # cut off time to stationarity
        self.timeseries = timeseries[n_stationary:]

        # safe to ./data/simulation if file_name given
        if file_name is not None:
            df = pd.DataFrame(self.timeseries[self.order:])
            df.index = range(n_steps)
            df.to_csv(f'./data/simulations/{file_name}.csv')

        return self.timeseries[self.order:]


    def directed_information(self, x, y, z=[]):
        """
        Calculate I(X -> Y || Z), the Directed Information (DI) from variable X to Y
        causally conditioned on a set of variables Z.

        Comparing two models, one given X and Z another one only given Z:
        Y_t = a * Y_t-1 + b * Z_t-1 + c * X_t-1 + ... + N_t
        Y_t = a' * Y_t-1 + b' * Z_t-1 + ... + N_t'

        I(X -> Y || Z) = H(Y || Z) - H(Y || X, Z)
                       = log (std(N_t') / std(N_t))
        where H(Y || X, Z) = 0.5 * log (2 * pi * e * std(N_t)^2)

        Parameters:
        - x (int): Index of the source variable X.
        - y (int): Index of the target variable Y.
        - z (list): List of indices of variables in the set Z.

        Returns:
        - di (float): The calculated Directed Information.
        """

        steps = len(self.timeseries) - self.order
        # calculated with values of real simulation
        y_given_xz = np.zeros(steps)
        y_given_z = np.zeros(steps)

        yxz = [y] + [x] + z
        yz = [y] + z

        if x == y:
            yxz = [y] + z
            yz = z

        for t in range(steps):
            sum_y_given_xz = 0
            sum_y_given_z = 0

            for i in range(self.order):
                sum_y_given_xz += self.coefficients[i, y, yxz] @ self.timeseries[self.order - 1 + t - i, yxz]
                sum_y_given_z += self.coefficients[i, y, yz] @ self.timeseries[self.order - 1 + t - i, yz]

            y_given_xz[t] = sum_y_given_xz
            y_given_z[t] = sum_y_given_z

        residuals_y_given_xz = self.timeseries[self.order:, y] - y_given_xz
        residuals_y_given_z = self.timeseries[self.order:, y] - y_given_z

        std_residuals_given_xz = np.std(residuals_y_given_xz)
        std_residuals_given_z = np.std(residuals_y_given_z)

        di = np.log(std_residuals_given_z / std_residuals_given_xz)

        return di


    # TODO: subset selection?
    def directed_information_graph(self, plot=False, threshold=0.05, subset_selection=None):
        """"
        Compute directed information (DI) between all variables and plot results
        in a Direct Information Graph (DIG).

        Parameters:
        - threshold (float): The threshold for DI in graph (default is 0.05).
        - plot (boolean): Whether the graph should be plotted or not.
        - subset_selection (object): Subset selection policy used to determine set causally conditioned on.

        Returns:
        - di_matrix (numpy.ndarray) or graphviz.Digraph: A matrix containing DI values between all variables
        or the graphical representation of the DIG, depending on the parameter plot.
        """

        for i in range(self.m):
            for j in range(self.m):
                z = [r for r in range(self.m) if r != i and r != j]
                self.di_matrix[i, j] = self.directed_information(i, j, z)

        if plot:
            labels = [f'X{i}' for i in range(self.m)]
            graph = utils.plot_directed_graph('dig', np.array([self.di_matrix]), labels, threshold)

        return graph if plot else self.di_matrix


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


class NVAR:
    def __init__(self, m, order=1):
        """
        Initialize a Non-Linear Vector Autoregressive NVAR(p) model of form
        X_t = f(X_t-1, X_t-2, ..., X_t-p) + N_t

        Parameters:
        - m (int): The number of processes (variables).
        - order (int): The order of the VAR model (default is 1).

        Returns:
        - None
        """

        self.m = m
        self.order = order
        self.functions = []
        self.functions_str = []
        self.generated = False
        self.timeseries = []
        self.di_matrix = np.zeros((m, m))


    def generate(self, max_predictors):
        """
        Generate random functions for the VAR model while ensuring stability

        Parameters:
        - max_predictors (int): Maximum number of predictor variables in the function 
        for each variable in the model.

        Returns:
        - None
        """
        self.functions_str = np.full((self.order, self.m, self.m), '', dtype='<U10')
        self.functions = np.full((self.order, self.m, self.m), lambda x: 0)
        # assign functions randomly across all orders and variables
        for var in range(self.m):
            n_predictors = np.random.randint(0, max_predictors + 1)
            all_pairs = [(o, d) for o in range(self.order) for d in range(self.m)]
            np.random.shuffle(all_pairs)

            for (random_order, random_var) in all_pairs[:n_predictors]:
                string, fun = self.random_function()
                self.functions_str[random_order, var, random_var] = string
                self.functions[random_order, var, random_var] = fun

        self.generated = True


    def random_function(self):
        """
        Generate a random function for the NVAR model.

        Returns:
        - Tuple: A tuple containing a string representation of the function and the
          corresponding function itself.
        """
        n_functions = 6
        n = np.random.randint(1, n_functions + 1)
        # linear / coefficient
        if n == 1:
            c = np.random.normal(scale=1 / (self.order * self.m))
            def fun(x): return c * x
            s = f'{c:.3f}'
            return (s, fun)
        # log
        elif n == 2:
            s = 'log'
            def fun(x): return np.log(abs(x)) if abs(x) != 0 else 0
            return (s, fun)
        elif n == 3:
            s = 'sigm'
            def fun(x): return 1 / (1 + np.exp(-x))
            return (s, fun)
        # tanh
        elif n == 4:
            s = 'tanh'
            return (s, np.tanh)
        # fractional power [0, 1)
        elif n == 5:
            p = np.random.random_sample()
            s = f'**{p:.3f}'
            def fun(x): return np.sign(x) * (np.abs(x) ** p)
            return (s, fun)
        # sin
        elif n == 6:
            s = 'sin'
            return (s, np.sin)


    def simulate(self, n_steps, file_name=None):
        """
        Simulate time series from the NVAR model. Noise ~N(0, 1/4)

        Parameters:
        - n_steps (int): The number of time steps to simulate.
        - file_name (str): The name of the file to save the simulated data (optional).

        Returns:
        - timeseries (numpy.ndarray): Simulated time series data with shape (m, n_steps).
        """
        if not self.generated:
            self.generate(self.m * self.order / 2)

        # extra steps to ensure stationarity of timeseries
        n_stationary = 300
        timeseries = np.zeros((n_stationary + n_steps + self.order, self.m))
        # first order values are random ~N(0,1)
        timeseries[:self.order, :] = np.random.randn(self.order, self.m)

        for t in range(self.order, n_steps + n_stationary + self.order):
            for var in range(self.m):
                sum = 0
                for order in range(self.order):
                    for dep in range(self.m):
                        sum += self.functions[order, var, dep](timeseries[t - 1 - order, dep])

                noise = np.random.normal(scale=0.25)
                timeseries[t, var] = sum + noise

        # cut off time to stationarity
        self.timeseries = timeseries[n_stationary:]

        # safe to ./data/simulation if file_name given
        if file_name is not None:
            df = pd.DataFrame(self.timeseries[self.order:])
            df.index = range(n_steps)
            df.to_csv(f'./data/simulations/{file_name}.csv')

        return self.timeseries[self.order:]


    def directed_information(self, x, y, z=[]):
        """
        Calculate I(X -> Y || Z), the Directed Information (DI) from variable X to Y
        causally conditioned on a set of variables Z.

        Comparing two models, one given X and Z another one only given Z:
        Y_t = f(Y_t-1) + g(Z_t-1) * h(X_t-1) + ... + N_t
        Y_t = f(Y_t-1) + g(Z_t-1) + ... + N_t'

        I(X -> Y || Z) = H(Y || Z) - H(Y || X, Z)
                       = log (std(N_t') / std(N_t))
        where H(Y || X, Z) = 0.5 * log (2 * pi * e * std(N_t)^2)

        Parameters:
        - x (int): Index of the source variable X.
        - y (int): Index of the target variable Y.
        - z (list): List of indices of variables in the set Z.

        Returns:
        - di (float): The calculated Directed Information.
        """
        steps = len(self.timeseries) - self.order
        # calculated with values of real simulation
        y_given_xz = np.zeros(steps)
        y_given_z = np.zeros(steps)

        yz = [y] + z

        if x == y:
            yz = z

        for t in range(steps):
            sum_y_given_xz = 0
            sum_y_given_z = 0
            for order in range(self.order):
                for i in yz:
                    sum_y_given_xz += self.functions[order, y, i](self.timeseries[self.order - 1 + t - order, i])
                    sum_y_given_z += self.functions[order, y, i](self.timeseries[self.order - 1 + t - order, i])

                # add x
                sum_y_given_xz += self.functions[order, y, x](self.timeseries[self.order - 1 + t - order, x])

            y_given_xz[t] = sum_y_given_xz
            y_given_z[t] = sum_y_given_z

        residuals_y_given_xz = self.timeseries[self.order:, y] - y_given_xz
        residuals_y_given_z = self.timeseries[self.order:, y] - y_given_z

        std_residuals_given_xz = np.std(residuals_y_given_xz)
        std_residuals_given_z = np.std(residuals_y_given_z)

        di = np.log(std_residuals_given_z / std_residuals_given_xz)

        return di


    # TODO: subset selection?
    def directed_information_graph(self, plot=False, threshold=0.05, subset_selection=None):
        """"
        Compute directed information (DI) between all variables and plot results
        in a Direct Information Graph (DIG).

        Parameters:
        - threshold (float): The threshold for DI in graph (default is 0.05).
        - plot (boolean): Whether the graph should be plotted or not.
        - subset_selection (object): Subset selection policy used to determine set causally conditioned on.

        Returns:
        - di_matrix (numpy.ndarray) or graphviz.Digraph: A matrix containing DI values between all variables
        or the graphical representation of the DIG, depending on the parameter plot.
        """

        for i in range(self.m):
            for j in range(self.m):
                z = [r for r in range(self.m) if r != i and r != j]
                self.di_matrix[i, j] = self.directed_information(i, j, z)

        if plot:
            labels = [f'X{i}' for i in range(self.m)]
            graph = utils.plot_directed_graph('dig', np.array([self.di_matrix]), labels, threshold)

        return graph if plot else self.di_matrix
