from scipy.linalg import solve_discrete_lyapunov
import numpy as np
import pandas as pd


# VAR(p): Vector Autoregressive Model of order p with m processes
# X_t = C_1 * X_t-1 + C_2 X_t-2 + ... + C_p * X_t-p + N_t
# with X_t = (X_1, ..., X_m)^T and m x m coefficient matrices C_i
class VAR:
    def __init__(self, m, order=1):
        self.m = m
        self.order = order
        self.coefficients = []
        self.generated = False


    # generate random coefficients ensuring stability
    # with a max. of max_predictor predictor variables for each variable
    def generate(self, max_predictors):
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


    def get_VAR1(self):
        if self.order == 1:
            return self.coefficients[0]

        # companion form: reducing VAR(p) to VAR(1)
        # X_t = F * X_t-1 + N_t
        F_upper = np.hstack(self.coefficients)
        F_lower = np.eye(self.m * (self.order - 1), self.m * self.order)
        F = np.vstack([F_upper, F_lower])
        return F


    def cov(self):
        F = self.get_VAR1()
        F_noise = np.zeros((self.m * self.order, self.m * self.order))
        F_noise[:self.m, :self.m] = np.eye(self.m)

        return solve_discrete_lyapunov(F, F_noise)[:self.m, :self.m]


    def __repr__(self):
        rep = f'VAR({self.order}): '
        rep += 'X_t = '
        for i in range(self.order):
            rep += f'C_{i + 1} * X_t-{i + 1} + '
        rep += 'N_t\n'

        for i in range(self.order):
            rep += f'C_{1 + i} = \n' + self.coefficients[i].__repr__() + '\n'

        return rep
