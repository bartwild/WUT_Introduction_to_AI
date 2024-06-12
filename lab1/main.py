from Grad import steepest_gradient_descent
from Newton import newton_method
from utils import plot_points, START_PT, BETA


def main():
    val_x, val_y, x_pts, y_pts, i = steepest_gradient_descent(START_PT, BETA, 500000, 1e-12)
    print(val_x, val_y)
    print("steps: " + str(i))
    plot_points(x_pts, y_pts, "Steepest Gradient Descent")
    val_x, val_y, x_pts, y_pts, i = newton_method(START_PT, BETA*35, 500000, 1e-12)
    print(val_x, val_y)
    print("steps: " + str(i))
    plot_points(x_pts, y_pts, "Newton Method")


if __name__ == "__main__":
    main()
