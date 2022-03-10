#!/usr/bin/env python

import casadi as ca
import math
import matplotlib.pyplot as plt
import numpy as np


def f(x, u):
    """The dynamical model for a differential drive.

    States: [[x], [y], [heading], [left velocity], [right velocity]]
    Inputs: [[left voltage], [right voltage]]

    Returns:
    dx/dt
    """
    width = 0.699  # m
    Kv_linear = 3.02  # V/(m/s)
    Ka_linear = 0.642  # V/(m/s^2)
    Kv_angular = 1.382  # V/(m/s)
    Ka_angular = 0.08495  # V/(m/s^2)

    A1 = 0.5 * (-(Kv_linear / Ka_linear + Kv_angular / Ka_angular))
    A2 = 0.5 * (-(Kv_linear / Ka_linear - Kv_angular / Ka_angular))
    B1 = 0.5 * (1 / Ka_linear + 1 / Ka_angular)
    B2 = 0.5 * (1 / Ka_linear - 1 / Ka_angular)

    A = np.array([[A1, A2], [A2, A1]])
    B = np.array([[B1, B2], [B2, B1]])

    v = (x[3] + x[4]) / 2

    return (
        ca.vertcat(
            v * ca.cos(x[2]),
            v * ca.sin(x[2]),
            (x[4] - x[3]) / width,
            A @ x[3:5],
        )
        + np.block([[np.zeros((3, 2))], [B]]) @ u
    )


def lerp(a, b, t):
    return a + t * (b - a)


def add_waypoint_constraints(opti, X, waypoints, N):
    """Add waypoint constraints.

    Keyword arguments:
    opti -- the optimizer
    X -- the state variables from the optimizer
    waypoints -- the list of waypoints
    N -- number of points per segment between waypoints
    """
    for i in range(len(waypoints)):
        k = i * N
        opti.subject_to(X[0, k] == waypoints[i][0])
        opti.subject_to(X[1, k] == waypoints[i][1])
        opti.subject_to(X[2, k] == waypoints[i][2])


def main():
    waypoints = [(0, 0, 0), (4.5, 3, 0), (4, 1, math.pi)]
    N_per_segment = 100
    N = N_per_segment * (len(waypoints) - 1)

    opti = ca.Opti()
    X = opti.variable(5, N + 1)
    U = opti.variable(2, N)

    Ts = []
    dts = []
    for i in range(len(waypoints) - 1):
        # Initial guess as linear interpolation between poses
        for j in range(N_per_segment):
            opti.set_initial(
                X[0, N_per_segment * i + j],
                lerp(waypoints[i][0], waypoints[i + 1][0], j / N_per_segment),
            )
            opti.set_initial(
                X[1, N_per_segment * i + j],
                lerp(waypoints[i][1], waypoints[i + 1][1], j / N_per_segment),
            )
            opti.set_initial(
                X[2, N_per_segment * i + j],
                lerp(waypoints[i][2], waypoints[i + 1][2], j / N_per_segment),
            )

        T = opti.variable()
        opti.subject_to(T >= 0)
        opti.set_initial(T, 1)
        Ts.append(T)

        dt = T / N_per_segment
        dts.append(dt)

    # Linear cost on time
    J = sum(Ts)

    # Quadratic cost on control input
    for k in range(N):
        R = np.diag(1 / np.square([12, 12]))
        J += U[:, k].T @ R @ U[:, k] * dts[int(k / N_per_segment)]

    opti.minimize(J)

    # Dynamics constraint
    for k in range(N):
        # RK4 integration
        dt = dts[int(k / N_per_segment)]
        k1 = f(X[:, k], U[:, k])
        k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
        k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
        k4 = f(X[:, k] + dt * k3, U[:, k])
        opti.subject_to(X[:, k + 1] == X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    # Constrain start and end velocities to zero
    opti.subject_to(X[3:5, 0] == np.zeros((2, 1)))
    opti.subject_to(X[3:5, -1] == np.zeros((2, 1)))

    # Require drivetrain always goes forward
    # opti.subject_to((X[3, :] + X[4, :]) / 2 >= 0)

    # Input constraint
    opti.subject_to(opti.bounded(-12, U, 12))

    add_waypoint_constraints(opti, X, waypoints, N_per_segment)

    opti.solver("ipopt")
    sol = opti.solve()

    # TODO: Resample states and inputs at 5 ms period

    print("dts = ", [sol.value(dts[k]) for k in range(len(dts))])
    print("Total time=", sum([sol.value(Ts[k]) for k in range(len(Ts))]))

    # Generate times for time domain plots
    ts = np.linspace(0, N_per_segment, (N_per_segment + 1)) * max(0, sol.value(dts[0]))
    for k in range(1, len(waypoints) - 1):
        ts = np.concatenate(
            (
                ts,
                ts[-1]
                + np.linspace(0, N_per_segment, N_per_segment) * sol.value(dts[k]),
            )
        )

    # Plot Y vs X
    plt.figure()
    plt.plot(sol.value(X[0, :]), sol.value(X[1, :]), label="Y vs X")
    plt.title("Y vs X")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()

    # Print total time
    sol_dts = []
    for k in range(len(waypoints) - 1):
        sol_dts.append(sol.value(Ts[k] / N_per_segment))

    # Plot X vs time
    plt.figure()
    plt.plot(ts, sol.value(sol.value(X[0, :])), label="X")
    plt.title("X vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("X (m)")
    plt.legend()

    # Plot Y vs time
    plt.figure()
    plt.plot(ts, sol.value(sol.value(X[1, :])), label="Y")
    plt.title("Y vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("Y (m)")
    plt.legend()

    # Plot heading vs time
    plt.figure()
    plt.plot(ts, sol.value(sol.value(X[2, :])), label="Heading")
    plt.title("Heading vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (rad)")
    plt.legend()

    # Plot wheel velocities vs time
    plt.figure()
    plt.plot(ts, sol.value(sol.value(X[3, :])), label="Left velocity")
    plt.plot(ts, sol.value(sol.value(X[4, :])), label="Right velocity")
    plt.title("Velocity vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()

    # Plot voltages vs time
    plt.figure()
    plt.plot(ts[:-1], sol.value(sol.value(U[0, :])), label="Left voltage")
    plt.plot(ts[:-1], sol.value(sol.value(U[1, :])), label="Right voltage")
    plt.title("Voltage vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
