from utils import grad, f
import time


def steepest_gradient_descent(PT, beta, iter_limit, epsilon):
    time_start = time.time()
    x_list = [PT[0]]
    y_list = [PT[1]]
    i = 0
    d = grad(x_list[i], y_list[i])
    eps = 1
    while i < iter_limit and eps >= epsilon:
        d = grad(x_list[i], y_list[i])
        x_list.append(x_list[i] - beta * d[0])
        y_list.append(y_list[i] - beta * d[1])
        i += 1
        eps = abs(f(x_list[i], y_list[i]) - f(x_list[i-1], y_list[i-1]))
    print("Steepest gradient descent time: " + str(time.time() - time_start) + " seconds")
    return x_list[-1], y_list[-1], x_list, y_list, i
