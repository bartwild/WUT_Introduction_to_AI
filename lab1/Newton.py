from utils import grad, inv_hessian, f
import numpy as np
import time


def newton_method(PT, beta, iter_limit, epsilon):
    time_start = time.time()
    x_list = [PT[0]]
    y_list = [PT[1]]
    i = 0
    d = inv_hessian(x_list[i], y_list[i]) @ grad(x_list[i], y_list[i])
    eps = 1
    while i < iter_limit and eps >= epsilon:
        d = inv_hessian(x_list[i], y_list[i]) @ grad(x_list[i], y_list[i])
        x_list.append(x_list[i] - beta * d[0])
        y_list.append(y_list[i] - beta * d[1])
        i += 1
        eps = abs(f(x_list[i], y_list[i]) - f(x_list[i-1], y_list[i-1]))
    print("Newton time: " + str(time.time() - time_start) + " seconds")
    print(np.linalg.eigvals([[12*(x_list[i]**2)+4*y_list[i]-42, 4 * x_list[i] + 4 * y_list[i]],
                            [4 * x_list[i] + 4 * y_list[i], 4*x_list[i]+12*(y_list[i]**2)-26]]))
    return x_list[-1], y_list[-1], x_list, y_list, i
