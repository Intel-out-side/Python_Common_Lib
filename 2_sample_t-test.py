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

        if result_shapiro[1] <= 0.05 and result_bartlett[1] <= 0.05:  # student t-test for 2-data of the equal variances
            result = stats.ttest(self.data[0], self.data[1], equal_var=True)
            print("student t-test: \n")
            print(result)
        elif result_shapiro[1] <= 0.05 and result_bartlett[1] > 0.05:  # Welch's t-test for non-equal variances
            result = stats.ttest(self.data[0], self.data[1], equal_var=False)
            print("Welch t-test: \n")
            print(result)
        else:
            quit("two data do not follow the normal distribution")

    def shapiro_wilk_test(self):
        """
        Validation for normality.
        :param args: data1, data2
        :return: W, p-value
        """

        if len(self.data[0]) == len(self.data[1]):
            tgt_data = self.data[0] - self.data[1]
        else:
            quit("Error: data 1 and 2 not the same length")

        result = stats.shapiro(tgt_data)
        return result

    def bartlett_test(self):
        """
        Validation for the equal variances.
        :param args: data1, data2
        :return: B, p-value
        """
        result = stats.bartlett(self.data[0], self.data[1])
        return result


if __name__ == '__main__':
    data1 = np.random.normal(50, 10, 1000)
    data2 = np.random.normal(40, 10, 1000)
    # generate the random data that follows normal distribution
    plt.hist(data1)
    plt.hist(data2)
    plt.show()

    t_test = T_test(data1, data2)
    t_test.shapiro_wilk_test()
    t_test.bartlett_test()
    sleep(3)
    t_test.t_test()
