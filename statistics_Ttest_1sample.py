from scipy.stats import *
import numpy as np


class Mu_1sample:

    def __init__(self, data, alpha=0.05, mu0=0.0):
        self.n = len(data)
        self.df = self.n - 1
        self.alpha = alpha
        self.mu0 = mu0
        self.data = data

    def t_test(self, mode=1):
        """
        Null Hypothesis Ho: mu = mu0

        :param mode:
        Alternative Hypothesis
        1)H1: mu != mu0 -> two-sided test
        2)H1: mu > mu0  -> one-sided test
        3)H1: mu < mu0  -> one-sided test
        :return:
        """
        V = np.var(self.data, ddof=1)
        t_0 = (np.average(self.data) - self.mu0)/np.sqrt(V/self.n)
        print("t_0: ", t_0)

        rv = t(df=self.df)

        if mode == 1:
            R = rv.isf(self.alpha/2)
            print("R: ", R)
            if abs(t_0) >= R:
                print("Ho rejected.")
            else:
                print("Ho not rejected")
        elif mode == 2:
            R = rv.isf(self.alpha)
            print("R: ", R)
            if t_0 >= R:
                print("Ho rejected")
            else:
                print("Ho not rejected")
        elif mode == 3:
            R = rv.ppf(self.alpha)
            print("R: ", R)
            if t_0 <= R:
                print("Ho rejected")
            else:
                print("Ho not rejected")

    def statistic_power(self, delta=0.5, mode=1):
        lambda_ = np.sqrt(self.n) * delta  # parameter of non-centrality

        rv1 = t(df=self.df)  # t distribution of df = n-1
        rv2 = nct(df=self.df, nc=lambda_)  # non-central t distribution of df=n-1, nc=lambda_

        if mode == 1:
            # approximation
            w = rv1.isf(self.alpha/2)
            power = norm.cdf((-1*w*(1-1/(4*self.df)) - lambda_) / np.sqrt(1 + w**2/(2*self.df))) + 1 - \
                    norm.cdf((w*(1-1/(4*self.df)) - lambda_) / np.sqrt(1 + w**2/(2*self.df)))
            print("power(approx.): ", power)

            # exact value
            power = rv2.cdf(-1*t.isf(self.alpha/2, df=self.df)) + \
                    rv2.sf(t.isf(self.alpha/2, df=self.df))
            print("power(exact): ", power)
        elif mode == 2:
            # approximation
            w = rv1.isf(self.alpha)
            power = 1 - norm.cdf((w*(1-1/(4*self.df)) - lambda_) / np.sqrt(1 + w**2/(2*self.df)))
            print("power(aprox.): ", power)

            # exact value
            power = rv2.sf(t.isf(self.alpha, df=self.df))
            print("power(exact): ", power)

        elif mode == 3:
            # approximation
            w = rv1.isf(self.alpha)
            power = norm.cdf((-1*w*(1-1/(4*self.df)) - lambda_) / np.sqrt(1 + w**2/(2*self.df)))
            print("power(approx.): ", power)

            # exact value
            power = rv2.cdf(-t.isf(self.alpha, df=self.df))
            print("power(exact): ", power)

class Sample_size_estimation:

    def __init__(self):
        pass

    def estimate(self, alpha=0.05, delta=0.5, required_power=0.90, mode=1):

        if mode == 1:
            n = ((norm.isf(alpha/2) - norm.isf(required_power)) / delta)**2 + 0.5 * norm.isf(alpha/2)**2

            n_ = np.floor(n)
            lambda_ = np.sqrt(n_) * delta
            rv = nct(df=n_-1, nc=lambda_)
            power = rv.cdf(-1*t.isf(alpha/2, df=n_-1)) + rv.sf(t.isf(alpha/2, df=n_-1))
            if power >= required_power:
                return n_

            n_ = np.ceil(n)
            lambda_ = np.sqrt(n_) * delta
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.cdf(-1 * t.isf(alpha/2, df=n_-1)) + rv.sf(t.isf(alpha/2, df=n_-1))
            if power >= required_power:
                return n_

        elif mode == 2:
            n = ((norm.isf(alpha) - norm.isf(required_power)) / delta)**2 + 0.5 * norm.isf(alpha)**2

            n_ = np.floor(n)
            lambda_ = np.sqrt(n_) * delta
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.sf(t.isf(alpha, df=n_ - 1))
            if power >= required_power:
                return n_

            n_ = np.ceil(n)
            lambda_ = np.sqrt(n_) * delta
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.sf(t.isf(alpha, df=n_ - 1))
            if power >= required_power:
                return n_

            return np.floor(n) + 2

        elif mode == 3:
            n = ((norm.isf(alpha) - norm.isf(required_power)) / delta) ** 2 + 0.5 * norm.isf(alpha) ** 2

            n_ = np.floor(n)
            lambda_ = np.sqrt(n_) * delta
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.cdf(-t.isf(alpha, df=n_ - 1))
            if power >= required_power:
                return n_

            n_ = np.ceil(n)
            lambda_ = np.sqrt(n_) * delta
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.cdf(-t.isf(alpha, df=n_ - 1))
            if power >= required_power:
                return n_

            return np.floor(n) + 2


if __name__ == '__main__':
    # data = [6.2, 4.8, 7.3, 5.5, 6.5, 4.9, 6.8, 7.9, 6.6, 7.3]
    # data = [10.8, 11.2, 9.7, 9.9, 12.0, 9.6, 10.5, 10.7, 10.1]
    data = [21, 19, 16, 19, 22, 18, 20, 21]
    test = Mu_1sample(data, alpha=0.05, mu0=20.0)
    test.t_test(mode=3)
    test.statistic_power(delta=-1.0, mode=3)

    n = Sample_size_estimation().estimate(delta=-1.5, required_power=0.95, mode=3)
    print(n)
