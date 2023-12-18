import numpy as np
import analysis.utils as utils


class VAR:
    def __init__(self, m, order=1):
        """
        Initialize a Vector Autoregressive VAR(p) model of form
        X_t = C_1 * X_t-1 + C_2 X_t-2 + ... + C_p * X_t-p + N_t

        Parameters:
        - m (int): The number of variables.
        - order (int): The order p of the VAR model (default is 1).

        Returns:
        - None
        """
        self.m = m
        self.order = order
        # coefficient matrices of VAR(p) reduced to companion form
        self.companion_matrix = []
        self.generated = False
        self.simulated = False
        self.timeseries = []
        self.di_matrix = np.zeros((m, m))


    def generate(self, max_coefficients, coefficient_threshold=0.1):
        """
        Generate random coefficients for the VAR model while ensuring
        stationarity. The coefficients are ~N(0, 1) and must be greater or
        equal to the coefficient threshold. Each variable will have a maximum
        of max_coefficients non-zero coefficients uniformly distributed across
        all orders and variables.
        If the order is one, the coefficient matrix will be scaled to have
        maximum eigenvalue of 0.9.

        Parameters:
        - max_coefficients (int): Maximum number of non-zero coefficients for
        each variable in the model.
        - coefficient_threshold (float): The lower bound for the absolute value
        of non-zero coefficients

        Returns:
        - None
        """
        while True:
            # coefficient matrix
            c = np.zeros((self.order, self.m, self.m))

            # assign coefficients randomly across all orders and variables
            for i in range(self.m):
                n_non_zero = np.random.randint(0, max_coefficients + 1)
                all_pairs = [(o, v) for o in range(self.order) for v in range(self.m)]
                np.random.shuffle(all_pairs)

                for (order, var) in all_pairs[:n_non_zero]:
                    c[order, i, var] = np.random.randn()

            # cut off coefficients below threshold
            abs_c = np.abs(c)
            c = np.where(abs_c >= coefficient_threshold, c, 0)

            # get coefficients in companion form: VAR(p) -> VAR(1)
            var1 = c[0] if self.order == 1 else utils.reduce_to_VAR1(c)

            if utils.is_stable(var1):
                self.companion_matrix = var1
                break

        self.simulated = False
        self.generated = True


    def simulate(self, steps, std_noise=0.25):
        """
        Simulate time series from the VAR model. Noise ~N(0, std_noise).

        Parameters:
        - steps (int): The number of time steps to simulate.
        - std_noise (float): Standard deviation of the noise (default is 0.25).

        Returns:
        - timeseries (numpy.ndarray): Simulated time series data with shape (m, steps).
        """
        assert self.generated, 'VAR model can not be simulated wihtout being generated first.'

        # extra steps to ensure stationarity of timeseries
        steps_stationary = 500
        timeseries = np.zeros((steps + steps_stationary + self.order, self.m))
        # first order values are random ~N(0,1)
        timeseries[:self.order, :] = np.random.randn(self.order, self.m)

        for t in range(self.order, steps + steps_stationary + self.order):
            # X_t = X_past * companion_matrix + noise
            past_values = timeseries[t - 1] if self.order == 1 else np.flip(timeseries[t - self.order:t], 0).flatten()
            values = self.companion_matrix @ past_values
            noise = np.random.normal(scale=std_noise, size=self.m)

            timeseries[t, :] = values[:self.m] + noise

        # cut off time to stationarity, but leave first order steps
        # to make reconstruction / calculation of exact DI possible
        self.timeseries = timeseries[steps_stationary:]
        self.simulated = True

        return self.timeseries[self.order:]


    def directed_information(self, x, y, z=[]):
        """
        Calculate I(X -> Y || Z), the Directed Information (DI) from variable X to Y
        causally conditioned on a set of variables Z.

        Comparing two models, one given X and Z another one only given Z:
        Y_t  = a  * Y_t-1 + b  * Z_t-1 + c * X_t-1 + ... + N_t
        Y_t' = a' * Y_t-1 + b' * Z_t-1 + ... + N_t'

        I(X -> Y || Z) = H(Y || Z) - H(Y || X, Z)
                       = log (std(N_t') / std(N_t))
        where H(Y || X, Z) = 0.5 * log (2 * pi * e * std(N_t)^2)

        Parameters:
        - x (int): Index of the source variable X.
        - y (int): Index of the target variable Y.
        - z (list): List of indices of variables in the set causally conditioned on.

        Returns:
        - di (float): The calculated Directed Information.
        """
        assert self.simulated, 'Data of the VAR model has not been simulated or does not correspond to current model.'

        steps = self.timeseries.shape[0] - self.order

        # calculated with values of simulation
        y_given_xz = np.zeros(steps)
        y_given_z = np.zeros(steps)

        yxz = [y] + [x] + z
        yz = [y] + z

        if x == y:
            yxz = [y] + z
            yz = z

        # reconstruct coefficients for each lag from companion matrix
        coefficients = self.get_coefficients()

        for t in range(steps):
            sum_y_given_xz = 0
            sum_y_given_z = 0

            for i in range(self.order):
                sum_y_given_xz += coefficients[i, y, yxz] @ self.timeseries[self.order - 1 + t - i, yxz]
                sum_y_given_z += coefficients[i, y, yz] @ self.timeseries[self.order - 1 + t - i, yz]

            y_given_xz[t] = sum_y_given_xz
            y_given_z[t] = sum_y_given_z

        residuals_y_given_xz = self.timeseries[self.order:, y] - y_given_xz
        residuals_y_given_z = self.timeseries[self.order:, y] - y_given_z

        std_residuals_given_xz = np.std(residuals_y_given_xz)
        std_residuals_given_z = np.std(residuals_y_given_z)

        di = np.log(std_residuals_given_z / std_residuals_given_xz)

        return di


    def directed_information_graph(self, threshold=0.05):
        """"
        Compute directed information (DI) between all variables and plot
        results in a Direct Information Graph (DIG).

        Parameters:
        - threshold (float): The threshold for edges in the DIG to be shown
        (default is 0.05).

        Returns:
        - di_matrix (numpy.ndarray): A matrix containing DI values for all
        variables.
        - plot (graphviz.Digraph): Visual representation of the DIG.
        """

        for i in range(self.m):
            for j in range(self.m):
                z = [r for r in range(self.m) if r != i and r != j]
                self.di_matrix[i, j] = self.directed_information(i, j, z)

        labels = [f'X{i}' for i in range(self.m)]
        plot = utils.plot_directed_graph('dig', self.di_matrix, labels, threshold)

        return self.di_matrix, plot


    def get_coefficients(self):
        """
        Reconstructs coefficients for each lag from 2D companion matrix of the
        VAR(p) model. The returned matrix has the shape (p x n_vars x n_vars).

        Parameters:
        - None

        Returns:
        - coefficients (numpy.ndarray): 3D array of coefficients for each lag.
        """
        coefficients = [self.companion_matrix[:self.m, self.m * i:self.m * (i + 1)] for i in range(self.order)]
        return np.array(coefficients)


    def draw(self):
        """
        Returns the plot for the VAR(p) model, showing which variable at what
        lag influences other variables at time t. Edges correspond to the
        coefficients.

        Parameters:
        - None

        Returns:
        - graphviz.Digraph: Visual representation of the VAR model.
        """
        labels = [f'X{i}' for i in range(self.m)]
        return utils.plot_directed_graph('var', self.get_coefficients(), labels)


    def __repr__(self):
        rep = f'VAR({self.order}): '
        rep += 'X_t = '
        for i in range(self.order):
            rep += f'A_{i + 1} * X_t-{i + 1} + '
        rep += 'N_t\n'

        coefficients = self.get_coefficients()
        for i in range(self.order):
            rep += f'A_{1 + i} = \n' + coefficients[i].__repr__() + '\n'

        return rep


class NVAR:
    def __init__(self, m, order=1):
        """
        Initialize a Non-Linear Vector Autoregressive NVAR(p) model of form
        X_t = f(X_t-1, X_t-2, ..., X_t-p) + N_t

        Parameters:
        - m (int): The number of processes (variables).
        - order (int): The order p of the NVAR model (default is 1).

        Returns:
        - None
        """
        self.m = m
        self.order = order
        self.functions = []
        self.functions_str = []
        self.generated = False
        self.simulated = False
        self.timeseries = []
        self.di_matrix = np.zeros((m, m))


    def generate(self, max_functions):
        """
        Generate a random NVAR(p) model, where each variable is calculated as
        the sum of functions of past values. Each function on its own ensures
        stationarity and is randomly drawn from a pool of linear and non-Linear
        funcitons. Each variable is calculated using between [0, max_functions]
        functions, which take the value of a variable of the past as argument.
        The argument is uniformly distributed across all variables and lags.

        Parameters:
        - max_functions (int): Maximum number of functions used to calculate
        each variable in the model.

        Returns:
        - None
        """
        self.functions_str = np.full((self.order, self.m, self.m), '', dtype='<U10')
        self.functions = np.full((self.order, self.m, self.m), lambda x: 0)
        # assign functions randomly across all orders and variables
        for var in range(self.m):
            n_predictors = np.random.randint(0, max_functions + 1)
            all_pairs = [(o, d) for o in range(self.order) for d in range(self.m)]
            np.random.shuffle(all_pairs)

            for (random_order, random_var) in all_pairs[:n_predictors]:
                string, fun = self.random_function()
                self.functions_str[random_order, var, random_var] = string
                self.functions[random_order, var, random_var] = fun

        self.generated = True
        self.simulated = False


    def random_function(self):
        """
        Return a random function for the NVAR model.

        Returns:
        - Tuple: A tuple containing a string representation of the function
        and the corresponding function itself.
        """
        n_functions = 7
        n = np.random.randint(1, n_functions + 1)

        # linear / coefficient
        if n == 1:
            c = np.random.normal(scale=1 / (self.order * self.m))
            def fun(x): return c * x
            s = f'{c:.3f}'
            return (s, fun)
        # logarithm
        elif n == 2:
            s = 'log'
            def fun(x): return np.log(abs(x)) if abs(x) != 0 else 0
            return (s, fun)
        # sigmoid function
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
        # cos
        elif n == 7:
            s = 'cos'
            return (s, np.cos)


    def simulate(self, steps, std_noise=0.25):
        """
        Simulate time series from the NVAR model. Noise ~N(0, std_noise)

        Parameters:
        - n_steps (int): The number of time steps to simulate.
        - std_noise (float): Standard deviation of the noise (default is 0.25).

        Returns:
        - timeseries (numpy.ndarray): Simulated time series data with shape (m, steps).
        """
        assert self.generated, 'NVAR model can not be simulated wihtout being generated first.'

        # extra steps to ensure stationarity of timeseries
        n_stationary = 500
        timeseries = np.zeros((n_stationary + steps + self.order, self.m))
        # first order values are random ~N(0,1)
        timeseries[:self.order, :] = np.random.randn(self.order, self.m)

        for t in range(self.order, steps + n_stationary + self.order):
            for var in range(self.m):
                sum = 0
                for order in range(self.order):
                    for dep in range(self.m):
                        sum += self.functions[order, var, dep](timeseries[t - 1 - order, dep])

                noise = np.random.normal(scale=0.25)
                timeseries[t, var] = sum + noise

        # cut off time to stationarity, but leave first order steps
        # to make reconstruction / calculation of exact DI possible
        self.timeseries = timeseries[n_stationary:]
        self.simulated = True

        return self.timeseries[self.order:]


    def directed_information(self, x, y, z=[]):
        """
        Calculate I(X -> Y || Z), the Directed Information (DI) from variable X to Y
        causally conditioned on a set of variables Z.

        Comparing two models, one given X and Z another one only given Z:
        Y_t  = f(Y_t-1) + g(Z_t-1) + h(X_t-1) + ... + N_t
        Y_t' = f(Y_t-1) + g(Z_t-1) + ... + N_t'

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
        assert self.simulated, 'Data of the VAR model has not been simulated or does not correspond to current model.'

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

            # use results of functions of each order and variable in yz
            for order in range(self.order):

                for i in yz:
                    add = self.functions[order, y, i](self.timeseries[self.order - 1 + t - order, i])
                    sum_y_given_xz += add
                    sum_y_given_z += add

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


    def directed_information_graph(self, threshold=0.05, subset_selection=None):
        """"
        Compute directed information (DI) between all variables and plot results
        in a Direct Information Graph (DIG).

        Parameters:
        - threshold (float): The threshold for DI in graph (default is 0.05).
        - subset_selection (object): Subset selection policy used to determine set causally conditioned on.

        Returns:
        - di_matrix (numpy.ndarray): A matrix containing DI values between for variables.
        - plot (graphviz.Digraph): Visual representation of DIG.

        """
        for i in range(self.m):
            for j in range(self.m):
                z = [r for r in range(self.m) if r != i and r != j]
                self.di_matrix[i, j] = self.directed_information(i, j, z)

        labels = [f'X{i}' for i in range(self.m)]
        plot = utils.plot_directed_graph('dig', self.di_matrix, labels, threshold)

        return self.di_matrix, plot


    def draw(self):
        """
        Returns the plot for the NVAR(p) model, showing which variable at what
        lag is used as parameter for what funtions to calculate other variables
        at time t.

        Parameters:
        - None

        Returns:
        - graphviz.Digraph: Visual representation of the VAR model.
        """
        labels = [f'X{i}' for i in range(self.m)]
        return utils.plot_directed_graph('nvar', self.functions_str, labels)
