import matplotlib.pyplot as plt
import numpy as np


def mu_r(r):
    if 0 <= r < 50:
        return 1
    elif 50 <= r < 150:
        return 1 - 0.01*(r-50)
    else:
        return 0


def mu_v(v):
    if 0 <= v < 20:
        return 1
    elif 20 <= v < 120:
        return 1 - 0.01*(v - 20)
    else:
        return 0


def func(center, x):
    if center-1 <= x < center:
        return 1 * (x - (center-1))
    elif center <= x < center+1:
        return -1 * (x - (center+1))


def main():
    plt.subplot(1, 3, 1)
    r = np.arange(0, 200, 0.01)
    y1 = np.array([mu_r(r[i]) for i in range(len(r))])
    plt.axhline(0, color='b', linestyle='--')
    plt.axhline(1, color='b', linestyle='--')
    plt.axvline(50, color='b', linestyle='--')
    plt.axvline(150, color='b', linestyle='--')
    plt.plot(r, y1, label="Membership Function for r", color='r', linewidth=4.0)
    plt.ylabel("Is the radius is small?", fontsize=18)
    plt.xlabel("Radius [m]", fontsize=18)
    plt.legend(fontsize=10)

    plt.subplot(1, 3, 2)
    v = np.arange(0, 200, 0.01)
    y2 = np.array([mu_v(v[i]) for i in range(len(v))])
    plt.axhline(0, color='b', linestyle='--')
    plt.axhline(1, color='b', linestyle='--')
    plt.axvline(20, color='b', linestyle='--')
    plt.axvline(120, color='b', linestyle='--')
    plt.plot(v, y2, label="Membership Function for v", color='r', linewidth=4.0)
    plt.ylabel("Is the velocity small?", fontsize=18)
    plt.xlabel("Velocity [m/s]", fontsize=18)
    plt.legend(fontsize=10)

    plt.subplot(1, 3, 3)
    # deceleration
    a = np.arange(-2, 0, 0.01)
    y2 = np.array([func(-1, a[i]) for i in range(len(a))])
    plt.plot(a, y2, label="Deceleration", color='b', linewidth=4.0)

    # keep velocity
    a = np.arange(-1, 1, 0.01)
    y2 = np.array([func(-0, a[i]) for i in range(len(a))])
    plt.plot(a, y2, label="Keep velocity", color='g', linewidth=4.0)

    # acceleration
    a = np.arange(0, 2, 0.01)
    y2 = np.array([func(1, a[i]) for i in range(len(a))])
    plt.plot(a, y2, label="Acceleration", color='r', linewidth=4.0)

    plt.axhline(0, color=(0, 0, 0), linestyle='--')
    plt.axhline(1, color=(0, 0, 0), linestyle='--')
    plt.axvline(0, color=(0, 0, 0), linestyle='--')
    plt.axvline(-1, color=(0, 0, 0), linestyle='--')
    plt.axvline(1, color=(0, 0, 0), linestyle='--')

    plt.legend(fontsize=10)
    plt.show()

main()