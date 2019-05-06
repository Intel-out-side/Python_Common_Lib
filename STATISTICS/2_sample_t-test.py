"""
This library is implemented to simplify t-test.
Note:
    1. Only for 2-sample comparison.
    2. No need to explicitly examine the homogeneity of variance
"""

from scipy import stats
import numpy as np
from time import sleep
import matplotlib.pyplot as plt


class T_test:

    def __init__(self, *args):
        self.data = args

    def t_test(self):
        result_shapiro = self.shapiro_wilk_test()
        result_bartlett = self.bartlett_test()

        if result_shapiro[1] > 0.05 and result_bartlett[1] > 0.05:  # student t-test for 2-data of the equal variances
            result = stats.ttest_ind(self.data[0], self.data[1], equal_var=True)
            print("student t-test: \n")
            print(result)
        elif result_shapiro[1] > 0.05 and result_bartlett[1] <= 0.05:  # Welch's t-test for non-equal variances
            result = stats.ttest_ind(self.data[0], self.data[1], equal_var=False)
            print("Welch t-test: \n")
            print(result)
        else:
            quit("two data do not follow the normal distribution")

    def shapiro_wilk_test(self):
        """
        Validation for normality.
        Null Hypothesis Ho: mu1 == mu2
        :param args: data1, data2
        :return: W, p-value
        """

        if len(self.data[0]) == len(self.data[1]):
            tgt_data = self.data[0] - self.data[1]
        else:
            quit("Error: data 1 and 2 not the same length")

        result = stats.shapiro(tgt_data)
        # print("Shapiro test: \n")
        # print(result)
        return result

    def bartlett_test(self):
        """
        Validation for the equal variances.
        Null Hypothesis Ho: mu1 == mu2
        :param args: data1, data2
        :return: B, p-value
        """
        result = stats.bartlett(self.data[0], self.data[1])
        # print("Bartlett test: \n")
        # print(result)
        return result


if __name__ == '__main__':
    boys = np.array([143.1, 140.9, 147.2, 139.8, 141.3, 150.7, 149.4, 145.6, 146.5, 148.5, 141.2, 136.5, 145.8, 148.1, 144.3])
    girls = np.array([138.7, 142.8, 150.3, 148.4, 141.7, 149.5, 156.5, 144.6, 144.4, 145.7, 148.3, 140.8, 146.2, 149.9, 144.1])
    # generate the random data that follows normal distribution
    plt.hist(boys)
    plt.hist(girls)
    plt.show()

    t_test = T_test(boys, girls)
    # t_test.shapiro_wilk_test()
    # t_test.bartlett_test()
    t_test.t_test()
