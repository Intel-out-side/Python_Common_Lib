from scipy.stats import *
import numpy as np
from statistics_Ttest_1sample import Mu_1sample
from statistics_Ttest_1sample import Sample_size_1sample


class Mu_2sample(Mu_1sample):

    def __init__(self, alpha=0.05, **kwargs):
        super().__init__(alpha, **kwargs)

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
    data2 = [5.3, 6.2, 5.9, 7.3, 8.4, 7.3, 6.9, 7.6, 8.5, 8.1, 6.7, 7.7]

    # data1 = [10.8, 11.2, 9.7, 9.9, 12.0, 9.6, 10.5, 10.7, 10.1]
    # data2 = [10.2, 10.1, 9.9, 8.2, 10.2, 9.4, 10.4, 10.0]

    # data1 = [21, 19, 16, 19, 22, 18, 20, 21]
    # data2 = [19, 22, 21, 22, 25, 19, 24, 23, 19, 22]
    test = Mu_2sample(alpha=0.05, data1=data1, data2=data2, delta=-1.0)
    rv1, rv2, lambda_, df, t_0 = test.get_parameters()
    test.t_test(rv1, t_0, mode=3)
    test.statistic_power(rv1, rv2, lambda_, df, mode=3)

    n_sample = Sample_size_2sample(alpha=0.05, delta=0.5, required_power=0.90)
    n = n_sample.n_2sample(mode=2)
    print(n)
    n = n_sample.estimate(n, lambda_func=n_sample.lambda_, mode=2)
    print(n)
