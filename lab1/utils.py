import matplotlib.pyplot as plt
import numpy as np

BETA = 0.01
START_PT = [2, 2]


def f(x, y):
    return x ** 4 + 2 * (x ** 2) * y - 21 * (x ** 2) + 2 * x * (y ** 2) - 14 * x + (y ** 4) - 13 * (y ** 2) - 22 * y + 170


def grad(x, y):
    return np.array([4*(x**3) + 4*x*y - 42*x - 14 + 2*(y**2), 2*(x**2) + 4*x*y + 4*(y**3) - 26*y - 22])


def inv_hessian(x, y):
    hessian = np.array([[12*(x**2)+4*y-42, 4 * x + 4 * y],
                        [4 * x + 4 * y, 4*x+12*(y**2)-26]])
    determinant = hessian[0][0] * hessian[1][1] - (hessian[1][0] * hessian[0][1])
    hessian_to_inv = np.array([[hessian[1][1], -hessian[1][0]], [-hessian[0][1], hessian[0][0]]])
    return hessian_to_inv/determinant


def plot_points(list_x, list_y, title):
    plt.plot(list_x, list_y, label="X and Y pos")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y", rotation=0)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.legend()
    plt.grid(True)
    plt.show()
