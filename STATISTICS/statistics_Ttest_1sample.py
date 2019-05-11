from scipy.stats import *
import numpy as np


class Mu_1sample:

    def __init__(self, alpha=0.05, **kwargs):
        """
        :param alpha: 0.05 or 0.01
        :param kwargs:
        mu0, delta
        for 1 sample t-test -> mu0, data
        """
        self.data = kwargs
        self.alpha = alpha

    def __t_0(self):
        mu0 = self.data['mu0']
        data1 = self.data['data1']
        n = len(data1)
        df = n - 1

        V = np.var(data1, ddof=1)
        t_0 = (np.average(data1) - mu0)/np.sqrt(V/n)
        print("t_0: ", t_0)
        return t_0

    def __rv1(self):
        df = len(self.data['data1']) - 1
        return t(df=df)

    def __rv2(self):
        df = len(self.data['data1']) - 1
        nc = self.__lambda_()
        return nct(df=df, nc=nc)

    def __lambda_(self):
        n = len(self.data['data1'])
        delta = self.data['delta']
        return np.sqrt(n) * delta

    def __df(self):
        return len(self.data['data1']) - 1

    def get_parameters(self):
        rv1 = self.__rv1()
        rv2 = self.__rv2()
        lambda_ = self.__lambda_()
        df = self.__df()
        t_0 = self.__t_0()

        return rv1, rv2, lambda_, df, t_0

    def t_test(self, rv, t_0, mode=1):
        """
        Null Hypothesis Ho: mu = mu0

        :param mode:
        Alternative Hypothesis
        1)H1: mu != mu0 -> two-sided test
        2)H1: mu > mu0  -> one-sided test
        3)H1: mu < mu0  -> one-sided test
        :return:
        """

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

    def statistic_power(self, rv1, rv2, lambda_, df, mode=1):

        if mode == 1:
            # approximation
            w = rv1.isf(self.alpha/2)
            power = norm.cdf((-1*w*(1-1/(4*df)) - lambda_) / np.sqrt(1 + w**2/(2*df))) + 1 - \
                    norm.cdf((w*(1-1/(4*df)) - lambda_) / np.sqrt(1 + w**2/(2*df)))
            print("power(approx.): ", power)

            # exact value
            power = rv2.cdf(-1*t.isf(self.alpha/2, df=df)) + \
                    rv2.sf(t.isf(self.alpha/2, df=df))
            print("power(exact): ", power)

        elif mode == 2:
            # approximation
            w = rv1.isf(self.alpha)
            power = 1 - norm.cdf((w*(1-1/(4*df)) - lambda_) / np.sqrt(1 + w**2/(2*df)))
            print("power(aprox.): ", power)

            # exact value
            power = rv2.sf(t.isf(self.alpha, df=df))
            print("power(exact): ", power)

        elif mode == 3:
            # approximation
            w = rv1.isf(self.alpha)
            power = norm.cdf((-1*w*(1-1/(4*df)) - lambda_) / np.sqrt(1 + w**2/(2*df)))
            print("power(approx.): ", power)

            # exact value
            power = rv2.cdf(-t.isf(self.alpha, df=df))
            print("power(exact): ", power)


class Sample_size_1sample:

    def __init__(self, alpha, delta, required_power):
        self.alpha = alpha
        self.delta = delta
        self.required_power = required_power

    def lambda_(self, n):
        return np.sqrt(n) * self.delta

    def n_1sample(self, mode=1):
        n = None
        if mode == 1:
            n = ((norm.isf(self.alpha / 2) - norm.isf(self.required_power)) / self.delta) ** 2 + 0.5 * norm.isf(self.alpha / 2) ** 2

        elif mode == 2:
            n = ((norm.isf(self.alpha) - norm.isf(self.required_power)) / self.delta) ** 2 + 0.5 * norm.isf(self.alpha) ** 2

        elif mode == 3:
            n = ((norm.isf(self.alpha) - norm.isf(self.required_power)) / self.delta) ** 2 + 0.5 * norm.isf(self.alpha) ** 2

        return n

    def estimate(self, n, lambda_func, mode=1):

        if mode == 1:
            n_ = np.floor(n)
            lambda_ = lambda_func(n_)
            rv = nct(df=n_-1, nc=lambda_)
            power = rv.cdf(-1*t.isf(self.alpha/2, df=n_-1)) + rv.sf(t.isf(self.alpha/2, df=n_-1))
            if power >= self.required_power:
                return n_

            n_ = np.ceil(n)
            lambda_ = lambda_func(n_)
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.cdf(-1 * t.isf(self.alpha/2, df=n_-1)) + rv.sf(t.isf(self.alpha/2, df=n_-1))
            if power >= self.required_power:
                return n_

            return np.floor(n) + 2

        elif mode == 2:
            n_ = np.floor(n)
            lambda_ = lambda_func(n_)
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.sf(t.isf(self.alpha, df=n_ - 1))
            if power >= self.required_power:
                return n_

            n_ = np.ceil(n)
            lambda_ = lambda_func(n_)
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.sf(t.isf(self.alpha, df=n_ - 1))
            if power >= self.required_power:
                return n_

            return np.floor(n) + 2

        elif mode == 3:
            n_ = np.floor(n)
            lambda_ = lambda_func(n_)
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.cdf(-t.isf(self.alpha, df=n_ - 1))
            if power >= self.required_power:
                return n_

            n_ = np.ceil(n)
            lambda_ = lambda_func(n_)
            rv = nct(df=n_ - 1, nc=lambda_)
            power = rv.cdf(-t.isf(self.alpha, df=n_ - 1))
            if power >= self.required_power:
                return n_

            return np.floor(n) + 2


if __name__ == '__main__':
    data1 = [6.2, 4.8, 7.3, 5.5, 6.5, 4.9, 6.8, 7.9, 6.6, 7.3]
    # data2 = [10.8, 11.2, 9.7, 9.9, 12.0, 9.6, 10.5, 10.7, 10.1]
    # data3 = [21, 19, 16, 19, 22, 18, 20, 21]
    test = Mu_1sample(alpha=0.05, mu0=5.0, data1=data1, delta=0.5)
    rv1, rv2, lambda_, df, t_0 = test.get_parameters()
    test.t_test(rv1, t_0, mode=1)
    test.statistic_power(rv1, rv2, lambda_, df, mode=1)
    # del test

    # n_sample = Sample_size_1sample(alpha=0.05, delta=0.8, required_power=0.95)
    # n = n_sample.n_1sample(mode=1)
    # n = n_sample.estimate(n, mode=1)
    # print(n)