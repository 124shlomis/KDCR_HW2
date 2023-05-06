import numpy as np
from math import pi, sin, cos, sqrt
import matplotlib.pyplot as plt

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
    c1 = L ** 2 - xa1 ** 2 - ya1 ** 2 - R ** 2
    c2 = L ** 2 - xa2 ** 2 - ya2 ** 2 - R ** 2

    # theta1 solution
    q[0] = np.arctan2(b1, a1) + np.arctan2(elbows[0] * sqrt(a1 ** 2 + b1 ** 2 - c1 ** 2), c1)
    # theta2 solution
    q[1] = np.arctan2(b2, a2) + np.arctan2(elbows[1] * sqrt(a2 ** 2 + b2 ** 2 - c2 ** 2), c2)

    return q


def draw_inv(x):
    # draws all possibilities of inverse kinematics for given x
    elbows_options = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    q_poss = []
    for count, elbow_poss in enumerate(elbows_options):
        q_poss.append(inverse_kin(x, elbow_poss))

    # tool coordinates
    phi = x[2] * pi / 180  # translate to rad
    x1 = x[0] + r * cos(phi + (pi / 3))
    y1 = x[1] + r * sin(phi + (pi / 3))
    x2 = x[0]
    y2 = x[1]
    x3 = x[0] + r * cos(phi)
    y3 = x[1] + r * sin(phi)
    # plot circle
    theta = np.linspace(0, 2 * np.pi, 100)
    r_rob = R
    x_circ = r_rob * np.cos(theta)
    y_circ = r_rob * np.sin(theta)

    plt.rcParams['axes.grid'] = True
    fig_width = 8
    fig_height = 8.5

    fig2, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height), tight_layout=True)
    fig2.suptitle("Joints values for different possibilities of inverse kinematics ", fontsize=16)

    tool2 = plt.Polygon([[x1, y1],
                         [x2, y2],
                         [x3, y3]], facecolor='gray', label='_nolegend_')
    axs[0, 0].plot(x_circ, y_circ, color='black', label='_nolegend_', linewidth=3)
    axs[0, 0].add_patch(tool2)
    axs[0, 0].plot([x1, R * cos(q_poss[0][0])], [y1, R * sin(q_poss[0][0])], color='blue', marker='o', label='theta1')
    axs[0, 0].plot([x2, R * cos(q_poss[0][1])], [y2, R * sin(q_poss[0][1])], color='green', marker='o', label='theta2')
    axs[0, 0].plot([x3, R * cos(q_poss[0][2])], [y3, R * sin(q_poss[0][2])], color='red', marker='o', label='d3')
    axs[0, 0].set(xlabel='x [m]', ylabel='y [m]')
    axs[0, 0].legend(['theta1', 'theta2', 'd3'])
    axs[0, 0].set_title('Config 1')

    tool3 = plt.Polygon([[x1, y1],
                         [x2, y2],
                         [x3, y3]], facecolor='gray', label='_nolegend_')
    axs[0, 1].plot(x_circ, y_circ, color='black', label='_nolegend_', linewidth=3)
    axs[0, 1].add_patch(tool3)
    axs[0, 1].plot([x1, R * cos(q_poss[1][0])], [y1, R * sin(q_poss[1][0])], color='blue', marker='o', label='theta1')
    axs[0, 1].plot([x2, R * cos(q_poss[1][1])], [y2, R * sin(q_poss[1][1])], color='green', marker='o', label='theta2')
    axs[0, 1].plot([x3, R * cos(q_poss[1][2])], [y3, R * sin(q_poss[1][2])], color='red', marker='o', label='d3')
    axs[0, 1].set(xlabel='x [m]', ylabel='y [m]')
    axs[0, 1].legend(['theta1', 'theta2', 'd3'])
    axs[0, 1].set_title('Config 2')

    tool4 = plt.Polygon([[x1, y1],
                         [x2, y2],
                         [x3, y3]], facecolor='gray', label='_nolegend_')
    axs[1, 0].plot(x_circ, y_circ, color='black', label='_nolegend_', linewidth=3)
    axs[1, 0].add_patch(tool4)
    axs[1, 0].plot([x1, R * cos(q_poss[2][0])], [y1, R * sin(q_poss[2][0])], color='blue', marker='o', label='theta1')
    axs[1, 0].plot([x2, R * cos(q_poss[2][1])], [y2, R * sin(q_poss[2][1])], color='green', marker='o', label='theta2')
    axs[1, 0].plot([x3, R * cos(q_poss[2][2])], [y3, R * sin(q_poss[2][2])], color='red', marker='o', label='d3')
    axs[1, 0].set(xlabel='x [m]', ylabel='y [m]')
    axs[1, 0].legend(['theta1', 'theta2', 'd3'])
    axs[1, 0].set_title('Config 3')

    tool5 = plt.Polygon([[x1, y1],
                         [x2, y2],
                         [x3, y3]], facecolor='gray', label='_nolegend_')
    axs[1, 1].plot(x_circ, y_circ, color='black', label='_nolegend_', linewidth=3)
    axs[1, 1].add_patch(tool5)
    axs[1, 1].plot([x1, R * cos(q_poss[3][0])], [y1, R * sin(q_poss[3][0])], color='blue', marker='o', label='theta1')
    axs[1, 1].plot([x2, R * cos(q_poss[3][1])], [y2, R * sin(q_poss[3][1])], color='green', marker='o', label='theta2')
    axs[1, 1].plot([x3, R * cos(q_poss[3][2])], [y3, R * sin(q_poss[3][2])], color='red', marker='o', label='d3')
    axs[1, 1].set(xlabel='x [m]', ylabel='y [m]')
    axs[1, 1].legend(['theta1', 'theta2', 'd3'])
    axs[1, 1].set_title('Config 4')

    fig2.tight_layout()
    plt.show()
    return


def main():
    # inverse kinematics:
    x = [-3, -1, 45]
    elbows = [-1, -1]
    q_sol = inverse_kin(x, elbows)
    print(round(q_sol[0], 4), round(q_sol[1], 4), round(q_sol[2], 4))
    draw_inv(x)


if __name__ == '__main__':
    main()
