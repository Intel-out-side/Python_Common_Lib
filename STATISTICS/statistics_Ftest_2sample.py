from scipy.stats import *
import numpy as np

class Sigma_2sample:

    def __init__(self, data1, data2, alpha=0.05):
        self.n1 = len(data1)
        self.n2 = len(data2)
        self.dfnum = self.n1 - 1  # df for numerator
        self.dfden = self.n2 - 1  # df for denominator
        self.alpha = alpha

        self.data1 = data1
        self.data2 = data2

    def F_test(self, mode=1):
        """
        Null Hypothesis: Ho: sigma1 = sigma2

        :param mode: Alternative Hypothesis
        mode1) sigma1 != sigma2
        mode2) sigma1 > sigma2
        mode3) sigma1 < sigma2
        :return:
        """

        V1 = np.var(self.data1, ddof=1)  # unbiased variance
        V2 = np.var(self.data2, ddof=1)  # unbiased variance
        print("V1:", V1, " V2: ", V2)
        F_0 = V1 / V2
        print("F_0: ", F_0)

        rv = f(dfn=self.dfnum, dfd=self.dfden)  # F distribution for df1, df2

        # mode1: two-sided test
        # mode2: one-sided test

        if mode == 1:
            R_right = rv.isf(self.alpha/2)  #F(df1, df2; a/2)
            R_left = rv.isf(1-self.alpha/2)  #F(df1, df2; 1-a/2)
            print("R_left: ", R_left)
            print("R_right: ", R_right)

            if F_0 <= R_left or R_right <= F_0:
                print("Ho rejected")
            else:
                print("Ho not rejected")

        elif mode == 2:
            R_right = rv.isf(self.alpha)
            print("R_right: ", R_right)

            if F_0 >= R_right:
                print("Ho rejected")
            else:
                print("Ho not rejected")

        elif mode == 3:
            R_left = rv.isf(1-self.alpha)
            print("R_left: ", R_left)

            if F_0 <= R_left:
                print("Ho rejected")
            else:
                print("Ho not rejected")

    def statistic_power(self, delta=0.5, mode=1):
        # F(df1, df2, 1-P) = 1/F(df2, df1, P)
        rv1 = f(dfn=self.dfnum, dfd=self.dfden)
        rv2 = f(dfn=self.dfden, dfd=self.dfnum)
        if mode == 1:
            w1 = rv1.isf(self.alpha/2)
            w2 = rv2.isf(self.alpha/2)
            power = f.sf(w1/(delta**2), dfn=self.dfnum, dfd=self.dfden) + \
                    f.cdf(1/(w2*delta**2), dfn=self.dfnum, dfd=self.dfden)
            print("power: ", power)

        elif mode == 2:
            w1 = rv1.isf(self.alpha)
            power = f.sf(w1/(delta**2), dfn=self.dfnum, dfd=self.dfden)
            print("power: ", power)

        elif mode == 3:
            w2 = rv2.isf(self.alpha)
            power = f.cdf(1/(w2*delta**2), dfn=self.dfnum, dfd=self.dfden)
            print("power: ", power)


class Sample_size_estimation:

    @staticmethod
    def estimate(delta1=1.0, delta2=0.5, required_power=0.90, alpha=0.05, mode=1):
        # n1 = n2 = n
        # implemented assuming that delta1 = delta2
        # delta1 > 1.0   delta2 < 1.0
        if mode == 1:
            n = 1 + ((norm.isf(alpha/2) - norm.isf(required_power))/np.log(delta1))**2

            n_ = np.floor(n)
            dfnum, dfden = n_ - 1, n_ - 1
            w1 = f.isf(alpha / 2, dfn=dfnum, dfd=dfden)
            w2 = f.isf(alpha / 2, dfn=dfnum, dfd=dfden)
            power = f.sf(w1 / (delta1 ** 2), dfn=dfnum, dfd=dfden) + \
                    f.cdf(1 / (w2 * delta1 ** 2), dfn=dfnum, dfd=dfden)
            if power >= required_power:
                n = n_

            n_ = np.ceil(n)
            dfnum, dfden = n_ - 1, n_ - 1
            w1 = f.isf(alpha / 2, dfn=dfnum, dfd=dfden)
            w2 = f.isf(alpha / 2, dfn=dfnum, dfd=dfden)
            power = f.sf(w1 / (delta1 ** 2), dfn=dfnum, dfd=dfden) + \
                    f.cdf(1 / (w2 * delta1 ** 2), dfn=dfnum, dfd=dfden)
            if power >= required_power:
                n = n_
            else:
                n = np.floor(n) + 2

            m = 1 + ((norm.isf(alpha/2) - norm.isf(required_power))/np.log(delta2))**2

            m_ = np.floor(m)
            dfnum, dfden = m_ - 1, m_ - 1
            w1 = f.isf(alpha / 2, dfn=dfnum, dfd=dfden)
            w2 = f.isf(alpha / 2, dfn=dfnum, dfd=dfden)
            power = f.sf(w1 / (delta1 ** 2), dfn=dfnum, dfd=dfden) + \
                    f.cdf(1 / (w2 * delta1 ** 2), dfn=dfnum, dfd=dfden)
            if power >= required_power:
                m = m_

            m_ = np.ceil(m)
            dfnum, dfden = m_ - 1, m_ - 1
            w1 = f.isf(alpha / 2, dfn=dfnum, dfd=dfden)
            w2 = f.isf(alpha / 2, dfn=dfnum, dfd=dfden)
            power = f.sf(w1 / (delta1 ** 2), dfn=dfnum, dfd=dfden) + \
                    f.cdf(1 / (w2 * delta1 ** 2), dfn=dfnum, dfd=dfden)
            if power >= required_power:
                m = m_
            else:
                m = np.floor(m) + 2

            if m < n:
                return n
            else:
                return m

        elif mode == 2:
            n = 1 + ((norm.isf(alpha) - norm.isf(required_power)) / np.log(delta1)) ** 2

            n_ = np.floor(n)
            dfnum, dfden = n_ - 1, n_ - 1
            w1 = f.isf(alpha, dfn=dfnum, dfd=dfden)
            power = f.sf(w1 / (delta1 ** 2), dfn=dfnum, dfd=dfden)
            if power >= required_power:
                return n_

            n_ = np.ceil(n)
            dfnum, dfden = n_ - 1, n_ - 1
            w1 = f.isf(alpha, dfn=dfnum, dfd=dfden)
            power = f.sf(w1 / (delta1 ** 2), dfn=dfnum, dfd=dfden)
            if power >= required_power:
                return n_

            return np.floor(n) + 2

        elif mode == 3:
            n = 1 + ((norm.isf(alpha) - norm.isf(required_power)) / np.log(delta2)) ** 2

            n_ = np.floor(n)
            dfnum, dfden = n_ - 1, n_ - 1
            w2 = f.isf(alpha, dfn=dfnum, dfd=dfden)
            power = f.cdf(1 / (w2 * delta2 ** 2), dfn=dfnum, dfd=dfden)
            if power >= required_power:
                return n_

            n_ = np.ceil(n)
            dfnum, dfden = n_ - 1, n_ - 1
            w2 = f.isf(alpha, dfn=dfnum, dfd=dfden)
            power = f.cdf(1 / (w2 * delta2 ** 2), dfn=dfnum, dfd=dfden)
            if power >= required_power:
                return n_
            else:
                return np.floor(n) + 2

if __name__ == '__main__':
    # data1 = [4.2, 2.8, 5.3, 3.5, 4.5, 2.9, 4.8, 5.9, 4.6, 5.3]
    # data2 = [7.3, 6.9, 7.3, 8.4, 8.5, 8.1, 5.3, 7.6, 6.2, 5.9, 6.7]

    # data1 = [13.8, 10.2, 8.7, 7.9, 13.0, 11.6, 10.5, 8.7, 7.1]
    # data2 = [10.3, 10.0, 9.8, 8.8, 10.5, 9.4, 10.6, 10.1]

    data1 = [21, 19, 16, 19, 22, 18, 20, 21]
    data2 = [19, 22, 21, 22, 25, 19, 24, 23, 19, 22]

    test = Sigma_2sample(data1, data2)
    test.F_test(mode=3)
    test.statistic_power(delta=1/3, mode=3)

    n = Sample_size_estimation().estimate(delta2=0.25, required_power=0.95, mode=3)
    print(n)
