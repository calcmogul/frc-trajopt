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


def plot_states(times, states, labels, units):
    """Plots states in time domain with one figure per unit.

    Keyword arguments:
    times -- list of times
    states -- matrix of states (states x times)
    labels -- list of state label strings
    units -- list of state unit strings
    """
    # Build mapping from unit to states that have that unit
    unit_to_data = {}
    for i, unit in enumerate(units):
        try:
            unit_to_data[unit].append(i)
        except KeyError:
            unit_to_data[unit] = [i]

    for unit, indices in unit_to_data.items():
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel(unit)
        for i in indices:
            if len(indices) > 1:
                plt.plot(times, states[i, :], label=labels[i])
            else:
                plt.plot(times, states[i, :])
        if len(indices) > 1:
            plt.legend()


def plot_inputs(times, inputs, labels, units):
    """Plots inputs in time domain with one figure per unit.

    Keyword arguments:
    times -- list of times
    inputs -- matrix of inputs (inputs x times)
    input_labels -- list of input label strings
    input_units -- list of input unit strings
    """
    # Build mapping from unit to inputs that have that unit
    unit_to_data = {}
    for i, unit in enumerate(units):
        try:
            unit_to_data[unit].append(i)
        except KeyError:
            unit_to_data[unit] = [i]

    for unit, indices in unit_to_data.items():
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel(unit)
        for i in indices:
            if len(indices) > 1:
                plt.plot(times, inputs[i, :], label=labels[i])
            else:
                plt.plot(times, inputs[i, :])
        if len(indices) > 1:
            plt.legend()


def main():
    waypoints = [(0, 0, 0), (4.5, 3, 0), (4, 1, math.pi)]
    q = 2
    r = [12, 12]
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
    J = q * sum(Ts)

    # Quadratic cost on control input
    for k in range(N):
        R = np.diag(1 / np.square(r))
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

    print("dts = ", [sol.value(dts[k]) for k in range(len(dts))])
    print("Total time=", sum([sol.value(Ts[k]) for k in range(len(Ts))]))

    # Generate times for time domain plots
    ts = [0]
    for k in range(N):
        ts.append(ts[-1] + sol.value(dts[int(k / N_per_segment)]))

    states = sol.value(X)
    inputs = sol.value(U)

    # TODO: Resample states and inputs at 5 ms period

    # Plot Y vs X
    plt.figure()
    plt.plot(states[0, :], states[1, :])
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plot_states(
        ts,
        states,
        ["X", "Y", "Heading", "Left velocity", "Right velocity"],
        ["X (m)", "Y (m)", "Heading (rad)", "Velocity (m/s)", "Velocity (m/s)"],
    )

    plot_inputs(
        ts[:-1],
        inputs,
        ["Left voltage", "Right voltage"],
        ["Voltage (V)", "Voltage (V)"],
    )

    plt.show()


if __name__ == "__main__":
    main()
