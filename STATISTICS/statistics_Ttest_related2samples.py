from scipy.stats import *
import numpy as np
from statistics_Ttest_1sample import Mu_1sample


class Mu_2sample_related(Mu_1sample):

    def __init__(self, alpha=0.05, **kwargs):
        super().__init__(alpha, **kwargs)

    def __t_0(self):
        data1 = np.array(self.data['data1'])
        data2 = np.array(self.data['data2'])
        n1 = len(data1)
        n2 = len(data2)
        if n1 != n2:
            exit("data1 and data2 not the same length")
        n = n1  # = n2
        df = n - 1
        d = data1 - data2

        Vd = np.var(d, ddof=1)
        t_0 = np.average(d)/np.sqrt(Vd/n)
        print(t_0)
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


if __name__ == '__main__':
    # data1 = [6.2, 4.8, 7.3, 5.5, 6.5, 4.9, 6.8, 7.9, 6.6, 7.3]
    # data2 = [5.3, 6.2, 5.9, 7.3, 8.4, 7.3, 6.9, 7.6, 8.5, 8.1]
    # data1 = [10.8, 11.2, 9.7, 9.9, 12.0, 9.6, 10.5, 10.7, 10.1]
    # data2 = [10.2, 10.1, 9.9, 8.2, 10.2, 9.4, 10.4, 10.0, 10.5]
    data1 = [19, 19, 21, 19, 22, 19, 20, 21]
    data2 = [21, 22, 16, 22, 25, 18, 24, 23]
    test = Mu_2sample_related(alpha=0.05, data1=data1, data2=data2, delta=-1.0)
    rv1, rv2, lambda_, df, t_0 = test.get_parameters()
    test.t_test(rv1, t_0, mode=3)
    test.statistic_power(rv1, rv2, lambda_, df, mode=3)
