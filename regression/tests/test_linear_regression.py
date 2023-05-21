from regression.linear.linear_regression import LinearRegression
from regression.model.linear_regression_data import LinearRegressionData
import math


# data source: https://www.cuemath.com/data/regression-coefficients/
def test_basic_data_set():
    x_array = [43, 21, 25, 42, 57, 59]
    y_array = [99, 65, 79, 75, 87, 81]
    data = LinearRegressionData(x_array, y_array)

    lr = LinearRegression()
    lr.train(data)
    lr.plot(data)

    expect_a = 0.39
    expect_b = 65.14
    assert math.isclose(lr.a, expect_a, abs_tol=0.01)
    assert math.isclose(lr.b, expect_b, abs_tol=0.01)
