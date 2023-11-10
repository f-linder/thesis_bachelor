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


    def generate(self, max_predictors):
        """
        Generate random coefficients for the VAR model while ensuring stability

        Parameters:
        - max_predictors (int): Maximum number of predictor variables for
        each variable in the model.

        Returns:
        - None
        """

        # maximum |eigenvalue| < 0.9 to guarantee stationarity
        max = 1
        while max > 0.9:
            self.coefficients = np.zeros((self.order, self.m, self.m))
            # assign coefficients randomly across all orders and variables
            for var in range(self.m):
                n_predictors = np.random.randint(0, max_predictors + 1)
                all_pairs = [(o, d) for o in range(self.order) for d in range(self.m)]
                np.random.shuffle(all_pairs)

                for (random_order, random_var) in all_pairs[:n_predictors]:
                    self.coefficients[random_order][var][random_var] = np.random.randn()

            # compute eigenvalues
            F = self.get_VAR1()
            eigenvalues = np.linalg.eigvals(F)
            max = np.max(np.abs(eigenvalues))

        self.generated = True


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

        timeseries = np.zeros((n_steps + self.order, self.m))
        # first order values are random ~N(0,1)
        timeseries[:self.order, :] = np.random.randn(self.order, self.m)

        # TODO: ask whether to generate more steps and only return
        # timeseries when stationarity reached
        for t in range(self.order, n_steps + self.order):
            sum = np.zeros(self.m)
            for i in range(self.order):
                sum += np.dot(self.coefficients[i], timeseries[t - 1 - i, :])

            noise = np.random.randn(self.m)

            timeseries[t, :] = sum + noise

        # cut randomly generated values
        timeseries = timeseries[self.order:]

        # safe to ./data/simulation if file_name given
        if file_name is not None:
            df = pd.DataFrame(timeseries)
            df.index = range(n_steps)
            df.to_csv(f'./data/simulations/{file_name}.csv')

        return timeseries.transpose()


    def cov(self):
        """
        Calculate the covariance matrix of the VAR model.

        Returns:
        - cov_matrix (numpy.ndarray): Covariance matrix of shape (m, m).
        """

        F = self.get_VAR1()
        F_noise = np.zeros((self.m * self.order, self.m * self.order))
        F_noise[:self.m, :self.m] = np.eye(self.m)

        return solve_discrete_lyapunov(F, F_noise)[:self.m, :self.m]


    # DI(X_i -> X_j || Z) = log (sd(N_t) / sd(N_t'))
    # one model with X_i and one without
    # X_t = F * X_t-1 + N_t
    # X_t = F' * X_t-1,' + N_t' where F' and X_t-1' do not include X_i
    # directed information as 
    def directed_information(self, x, y, z=[]):
        """
        Calculate I(X -> Y || Z), the Directed Information (DI) from variable X to Z
        causally conditioned on a set of variables Z.

        Parameters:
        - x (int): Index of the source variable X.
        - y (int): Index of the target variable Y.
        - z (list): List of indices of variables in the set Z.

        Returns:
        - di (float): The calculated Directed Information.
        """

        cov = self.cov()

        # sets of indices corresponding to processes
        xz = [x] + z
        yz = [y] + z
        xyz = [x] + [y] + z

        # covariance submatrices
        cov_xz = np.array([[cov[i, j] for j in xz] for i in xz])
        cov_yz = np.array([[cov[i, j] for j in yz] for i in yz])
        cov_xyz = np.array([[cov[i, j] for j in xyz] for i in xyz])
        cov_z = np.array([[cov[i, j] for j in z] for i in z])

        h_xz = 0.5 * np.log((2 * np.pi * np.e)**len(xz) * np.linalg.det(cov_xz))
        h_yz = 0.5 * np.log((2 * np.pi * np.e)**len(yz) * np.linalg.det(cov_yz))
        h_xyz = 0.5 * np.log((2 * np.pi * np.e)**len(xyz) * np.linalg.det(cov_xyz))
        h_z = 0.5 * np.log((2 * np.pi * np.e)**len(z) * np.linalg.det(cov_z)) if z != [] else 0

        print(f'H(X,Z)=={h_xz}, H(Y,Z)={h_yz}, H(X,Y,Z)={h_xyz}, H(Z)={h_z}')

        di = h_xz + h_yz - h_xyz - h_z
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
