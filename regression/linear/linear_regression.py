import matplotlib.pyplot as plt
from regression.model.linear_regression_data import LinearRegressionData


class LinearRegression:
    def __init__(self):
        self.a = None
        self.b = None

    def train(self, data: LinearRegressionData):
        # Calculate coefficients: https://www.cuemath.com/data/regression-coefficients/
        n = len(data.x_array)
        x_array = data.x_array
        y_array = data.y_array

        xy_array = [x * y for x, y in zip(x_array, y_array)]
        xx_array = [x * x for x in x_array]

        self.a = (n * sum(xy_array) - sum(x_array) * sum(y_array)) / (n * sum(xx_array) - sum(x_array) ** 2)
        self.b = (sum(y_array) * sum(xx_array) - sum(x_array) * sum(xy_array)) / (n * sum(xx_array) - sum(x_array) ** 2)

    def predict(self, x):
        if self.a is None or self.b is None:
            raise Exception("Must train first")

        return self.a * x + self.b

    def plot(self, data):
        if self.a is None or self.b is None:
            raise Exception("Must train first")

        x_min = min(data.x_array)
        x_max = max(data.x_array)

        y_min = self.a * x_min + self.b
        y_max = self.a * x_max + self.b

        # Create a figure and axis
        plt.figure()

        # Plot the line
        plt.plot([x_min, x_max], [y_min, y_max])
        plt.scatter(data.x_array, data.y_array)

        # Show the plot
        plt.show()
