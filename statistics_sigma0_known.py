from scipy.stats import *
import numpy as np


class Sigma0_Known:

    def __init__(self, data, alpha=0.05, sigma0=1.0, mu0=0.0):
        self.n = len(data)
        self.df = self.n - 1
        self.alpha = alpha
        self.sigma0 = sigma0
        self.mu0 = mu0
        self.data = data

    def statistic_test(self, mode=1):
        """
        Null Hypothesis Ho: mu = mu0

        :param mode:
        Alternative Hypothesis
        1)H1: mu != mu0
        2)H1: mu > mu0
        3)H1: mu < mu0

        :return:
        """
        u_0 = (np.average(self.data) - self.mu0)/np.sqrt((self.sigma0**2/self.n))
        print("u_0: ", u_0)

        # mode = 1: two-sided test
        # mode = 2, 3: one-sided test

        if mode == 1:
            u = norm.isf(self.alpha/2, loc=0, scale=1)
            print("u: ", u)
            if abs(u_0) >= u:
                print("Ho rejected. mu not equals mu_0")
            else:
                print("Ho not rejected")

        elif mode == 2:
            u = norm.isf(self.alpha, loc=0, scale=1)
            print("u: ", u)
            if u <= u_0:
                print("Ho rejected. mu not equals mu_0")
            else:
                print("Ho not rejected")

        elif mode == 3:
            u = norm.ppf(self.alpha, loc=0, scale=1)
            print("u: ", u)
            if u_0 <= u:
                print("Ho rejected. mu not equals mu_0")
            else:
                print("Ho not rejected")

    def statistic_power(self, delta=0.5, mode=1):
        """
        power = 1 - beta

        :param delta:
        :return:
        """
        if mode == 1:
            power = norm.cdf(-norm.isf(self.alpha/2)-np.sqrt(self.n)*delta) + \
                    norm.sf(norm.isf(self.alpha/2)-np.sqrt(self.n)*delta)
            print("power: ", power)
        elif mode == 2:
            power = norm.sf(norm.isf(self.alpha)-np.sqrt(self.n)*delta)
            print("power: ", power)
        elif mode == 3:
            power = norm.cdf(-norm.isf(self.alpha)-np.sqrt(self.n)*delta)
            print("power: ", power)


class Sample_size_estimation:

    def __init__(self):
        pass

    def estimate(self, alpha=0.05, delta0=0.5, required_power=0.90, mode=1):
        if mode == 1:
            n = ((norm.isf(alpha/2) - norm.isf(required_power))/delta0)**2

            n_ = np.floor(n)
            if norm.cdf(-norm.isf(alpha/2)-np.sqrt(n_)*delta0)+norm.sf(norm.isf(alpha/2)-np.sqrt(n_)*delta0) >= required_power:
                return n_

            n_ = np.ceil(n)
            if norm.cdf(-norm.isf(alpha/2)-np.sqrt(n_)*delta0)+norm.sf(norm.isf(alpha/2)-np.sqrt(n_)*delta0) >= required_power:
                return n_

        elif mode == 2:
            n =((norm.isf(alpha) - norm.isf(required_power))/delta0)**2

            n_ = np.floor(n)
            if norm.sf(norm.isf(alpha)-np.sqrt(n_)*delta0) >= required_power:
                return n_

            n_ = np.ceil(n)
            if norm.sf(-norm.isf(alpha)-np.sqrt(n_)*delta0) >= required_power:
                return n_

        elif mode == 3:
            n = ((norm.isf(alpha) - norm.isf(required_power)) / delta0) ** 2

            n_ = np.floor(n)
            if norm.cdf(-norm.isf(alpha)-np.sqrt(n_)*delta0) >= required_power:
                return n_

            n_ = np.ceil(n)
            if norm.cdf(-norm.isf(alpha)-np.sqrt(n_)*delta0) >= required_power:
                return n_


if __name__ == '__main__':
    data = [21, 19, 16, 19, 22, 18, 20, 21]
    test = Sigma0_Known(data, sigma0=2.0, mu0=20.0)
    test.statistic_test(mode=3)
    test.statistic_power(delta=-1.0, mode=3)

    n = Sample_size_estimation().estimate(alpha=0.05, delta0=-1.5, required_power=0.95, mode=3)
    print(n)

