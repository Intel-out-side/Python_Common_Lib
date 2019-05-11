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


class Mu_2sample(Mu_1sample):

    def __df(self):
        data1 = self.data['data1']
        data2 = self.data['data2']
        n1 = len(data1)
        n2 = len(data2)
        df = n1 + n2 - 2
        return df

    def __t_0(self):
        data1 = self.data['data1']
        data2 = self.data['data2']
        n1 = len(data1)
        n2 = len(data2)
        df = n1 + n2 - 2

        S1 = np.sum([(data1[i] - np.average(data1)) ** 2 for i in range(n1)])
        S2 = np.sum([(data2[i] - np.average(data2)) ** 2 for i in range(n2)])
        V = (S1 + S2) / (n1 + n2 - 2)

        t_0 = (np.average(data1) - np.average(data2)) / np.sqrt(V * (1 / n1 + 1 / n2))
        print("t_0: ", t_0)
        return t_0

    def __rv1(self):
        return t(df=self.__df())

    def __lambda_(self):
        data1 = self.data['data1']
        data2 = self.data['data2']
        n1 = len(data1)
        n2 = len(data2)
        delta = self.data['delta']
        return np.sqrt(n1*n2/(n1+n2))*delta

    def __rv2(self):
        df = self.__df()
        return nct(df=df, nc=self.__lambda_())

    def get_parameters(self):
        rv1 = self.__rv1()
        rv2 = self.__rv2()
        lambda_ = self.__lambda_()
        t_0 = self.__t_0()
        df = self.__df()

        return rv1, rv2, lambda_, df, t_0


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


class Sample_size_2sample(Sample_size_1sample):

    def __init__(self, alpha, delta, required_power):
        super().__init__(alpha, delta, required_power)

    def lambda_(self, n):
        return np.sqrt(n*n/(n+n)) * self.delta

    def n_2sample(self, mode=1):
        n = None
        if mode == 1:
            n = 2*((norm.isf(self.alpha / 2) - norm.isf(self.required_power)) / self.delta) ** 2 + 0.25 * norm.isf(self.alpha / 2) ** 2

        elif mode == 2:
            n = 2*((norm.isf(self.alpha) - norm.isf(self.required_power)) / self.delta) ** 2 + 0.25 * norm.isf(self.alpha) ** 2

        elif mode == 3:
            n = 2*((norm.isf(self.alpha) - norm.isf(self.required_power)) / self.delta) ** 2 + 0.25 * norm.isf(self.alpha) ** 2

        return n


if __name__ == '__main__':
    data1 = [6.2, 4.8, 7.3, 5.5, 6.5, 4.9, 6.8, 7.9, 6.6, 7.3]
    # data2 = [10.8, 11.2, 9.7, 9.9, 12.0, 9.6, 10.5, 10.7, 10.1]
    # data3 = [21, 19, 16, 19, 22, 18, 20, 21]
    test = Mu_1sample(alpha=0.05, mu0=5.0, data1=data1, delta=0.5)
    rv1, rv2, lambda_, df, t_0 = test.get_parameters()
    test.t_test(rv1, t_0, mode=1)
    test.statistic_power(rv1, rv2, lambda_, df, mode=1)
    del test

    # data1 = [6.2, 4.8, 7.3, 5.5, 6.5, 4.9, 6.8, 7.9, 6.6, 7.3]
    # data2 = [5.3, 6.2, 5.9, 7.3, 8.4, 7.3, 6.9, 7.6, 8.5, 8.1, 6.7, 7.7]
    # sample datas for two-sided test

    # data1 = [10.8, 11.2, 9.7, 9.9, 12.0, 9.6, 10.5, 10.7, 10.1]
    # data2 = [10.2, 10.1, 9.9, 8.2, 10.2, 9.4, 10.4, 10.0]
    # sample datas for one-sided test mode 2

    data1 = [21, 19, 16, 19, 22, 18, 20, 21]
    data2 = [19, 22, 21, 22, 25, 19, 24, 23, 19, 22]
    # sample datas for one-sided test mode 3
    test = Mu_2sample(alpha=0.05, data1=data1, data2=data2, delta=-1.0)
    rv1, rv2, lambda_, df, t_0 = test.get_parameters()
    test.t_test(rv1, t_0, mode=3)
    test.statistic_power(rv1, rv2, lambda_, df, mode=3)

    # n_sample = Sample_size_1sample(alpha=0.05, delta=0.8, required_power=0.95)
    # n = n_sample.n_1sample(mode=1)
    # n = n_sample.estimate(n, mode=1)
    # print(n)

    n_sample = Sample_size_2sample(alpha=0.05, delta=0.5, required_power=0.90)
    n = n_sample.n_2sample(mode=2)
    print(n)
    n = n_sample.estimate(n, lambda_func=n_sample.lambda_, mode=2)
    print(n)
