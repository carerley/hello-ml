from regression.logistic.logistic_regression import LogisticRegression


def test_basic_data_set():
    x_array = list(range(-10, 11))
    print(x_array)

    lr = LogisticRegression(1, -5)
    lr.plot(x_array)
