#!/usr/bin/env python

import casadi as ca
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def f(
    x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The dynamical model for a differential drive.

    Keyword arguments:
    x -- state column vector: [x, y, heading, left velocity, right velocity]
    u -- input column vector: [left voltage, right voltage]

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


def plot_data(
    times: list[float],
    data: npt.NDArray[np.float64],
    labels: list[str],
    units: list[str],
) -> None:
    """Plots data (e.g., states, inputs) in time domain with one figure per
    unit.

    Keyword arguments:
    times -- list of times
    data -- matrix of data (states or inputs x times)
    labels -- list of data label strings
    units -- list of data unit strings
    """
    # Build mapping from unit to data that have that unit
    unit_to_data = {}
    for i, unit in enumerate(units):
        try:
            unit_to_data[unit].append(i)
        except KeyError:
            unit_to_data[unit] = [i]

    for unit, indices in unit_to_data.items():
        plt.figure()
        plt.title(f"{unit.split()[0]} vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel(unit)
        for i in indices:
            if len(indices) > 1:
                plt.plot(times, data[i, :], label=labels[i])
            else:
                plt.plot(times, data[i, :])
        if len(indices) > 1:
            plt.legend()


class Waypoint:
    def __init__(self, x: float, y: float, heading: float | None = None) -> None:
        self.x = x
        self.y = y
        self.heading = heading


class DifferentialDriveTrajectoryOptimizer:
    def __init__(self) -> None:
        self.opti = ca.Opti()
        self.waypoints = []  # List of Waypoints

    def add_pose(self, x: float, y: float, heading: float) -> None:
        self.waypoints.append(Waypoint(x, y, heading))

    def add_translation(self, x: float, y: float) -> None:
        self.waypoints.append(Waypoint(x, y))

    def optimize(self, q: float, r: list[float]):
        """Generate the optimal trajectory.

        Keyword arguments:
        q -- minimum time cost weight
        r -- list of the maximum allowed excursions of the control inputs from
             no actuation

        Returns:
        sol -- solution object
        times -- list of times in solution
        X --
        """
        num_segments = len(self.waypoints) - 1
        vars_per_segment = 100
        num_vars = vars_per_segment * num_segments

        X = self.opti.variable(5, num_vars + 1)
        U = self.opti.variable(2, num_vars)

        # Apply waypoint constraints
        for i, waypoint in enumerate(self.waypoints):
            self.opti.subject_to(X[0, i * vars_per_segment] == waypoint.x)
            self.opti.subject_to(X[1, i * vars_per_segment] == waypoint.y)
            if waypoint.heading is not None:
                self.opti.subject_to(X[2, i * vars_per_segment] == waypoint.heading)

        for segment in range(num_segments):
            waypoint = self.waypoints[segment]
            next_waypoint = self.waypoints[segment + 1]

            segment_start = segment * vars_per_segment

            # Set initial guess for poses as linear interpolation between
            # waypoints
            for i in range(vars_per_segment):
                self.opti.set_initial(
                    X[0, segment_start + i],
                    lerp(waypoint.x, next_waypoint.x, i / vars_per_segment),
                )
                self.opti.set_initial(
                    X[1, segment_start + i],
                    lerp(waypoint.y, next_waypoint.y, i / vars_per_segment),
                )
                self.opti.set_initial(
                    X[2, segment_start + i],
                    math.atan2(
                        next_waypoint.y - waypoint.y,
                        next_waypoint.x - waypoint.x,
                    ),
                )

        # Set up duration decision variables
        Ts = []
        dts = []
        for segment in range(num_segments):
            T = self.opti.variable()
            self.opti.subject_to(T >= 0)
            self.opti.set_initial(T, 1)
            Ts.append(T)

            dt = T / vars_per_segment
            dts.append(dt)

        # Linear cost on time
        J = q * sum(Ts)

        # Quadratic cost on control input
        for k in range(num_vars):
            R = np.diag(1 / np.square(r))
            J += U[:, k].T @ R @ U[:, k] * dts[int(k / vars_per_segment)]

        self.opti.minimize(J)

        # Dynamics constraint
        for k in range(num_vars):
            # RK4 integration
            dt = dts[int(k / vars_per_segment)]
            k1 = f(X[:, k], U[:, k])
            k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
            k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
            k4 = f(X[:, k] + dt * k3, U[:, k])
            self.opti.subject_to(
                X[:, k + 1] == X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            )

        # Constrain start and end velocities to zero
        self.opti.subject_to(X[3:5, 0] == np.zeros((2, 1)))
        self.opti.subject_to(X[3:5, -1] == np.zeros((2, 1)))

        # Require drivetrain always goes forward
        # self.opti.subject_to((X[3, :] + X[4, :]) / 2 >= 0)

        # Input constraint
        self.opti.subject_to(self.opti.bounded(-r[0], U[0, :], r[0]))
        self.opti.subject_to(self.opti.bounded(-r[1], U[1, :], r[1]))

        self.opti.solver("ipopt")
        sol = self.opti.solve()

        # Generate times for time domain plots
        times = [0]
        for k in range(num_vars):
            times.append(times[-1] + sol.value(dts[int(k / vars_per_segment)]))

        print("dts = ", [sol.value(dts[k]) for k in range(len(dts))])
        print("Total time=", times[-1])

        # TODO: Resample states and inputs at 5 ms period

        return times, sol.value(X), sol.value(U)


def main():
    traj = DifferentialDriveTrajectoryOptimizer()
    traj.add_pose(0, 0, 0)
    traj.add_translation(4.5, 3)
    traj.add_pose(4, 1, -math.pi)
    times, states, inputs = traj.optimize(2, [12, 12])

    # Plot Y vs X
    plt.figure()
    plt.title("Y vs X")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.plot(states[0, :], states[1, :])

    plot_data(
        times,
        states,
        ["X", "Y", "Heading", "Left velocity", "Right velocity"],
        ["X (m)", "Y (m)", "Heading (rad)", "Velocity (m/s)", "Velocity (m/s)"],
    )
    plot_data(
        times[:-1],
        inputs,
        ["Left voltage", "Right voltage"],
        ["Voltage (V)", "Voltage (V)"],
    )

    plt.show()


if __name__ == "__main__":
    main()
