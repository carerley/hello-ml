import math

from regression.model.linear_regression_data import LinearRegressionData
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, a, b):
        # y = 1 / (1 + e^-(ax+b))
        self.a = a
        self.b = b

    def plot(self, x_array):
        y_array = [1 / (1 + math.e ** -(self.a * x + self.b)) for x in x_array]
        plt.figure()
        plt.plot(x_array, y_array)
        plt.show()
