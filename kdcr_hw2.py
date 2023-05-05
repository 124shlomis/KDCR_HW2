import numpy as np
from math import pi, sin, cos, sqrt

# Global constants:
L = 3.5
r = 2.0
R = 4.0
DEG2RAD = pi / 180


def inverse_kin(x, elbows):
    # calculate the inverse kinematics of the parallel robot
    # x = x[0], y = x[1], phi(deg) = x[2]
    # elb1 = elbows[0], elb2 = elbows[1]
    # q[0] = theta1, q[1] = theta 2 q[2] = theta3

    q = [0, 0, 0]
    phi = x[2] * DEG2RAD  # translate to rad

    xa1 = x[0] + r * cos(phi + pi / 3)
    ya1 = x[1] + r * sin(phi + pi / 3)
    xa2 = x[0]
    ya2 = x[1]
    xa3 = x[0] + r * cos(phi)
    ya3 = x[1] + r * sin(phi)

    # d3 solution
    q[2] = sqrt(xa3 ** 2 + (ya3 + R) ** 2)

    # shortcuts for atan2
    a1 = -2 * xa1 * R
    b1 = -2 * ya1 * R
    a2 = -2 * xa2 * R
    b2 = -2 * ya2 * R
    c = L ** 2 - x[0] ** 2 - x[1] ** 2 - R ** 2

    # theta1 solution
    q[0] = np.arctan2(b1, a1) + np.arctan2(elbows[0] * sqrt(a1 ** 2 + b1 ** 2 - c ** 2), c)
    # theta2 solution
    q[1] = np.arctan2(b2, a2) + np.arctan2(elbows[1] * sqrt(a2 ** 2 + b2 ** 2 - c ** 2), c)

    return q


def main():
    # inverse kinematics:
    x = [-3, 1, 45]
    elbows = [1, -1]
    q_sol = inverse_kin(x, elbows)
    print(q_sol)


if __name__ == '__main__':
    main()
