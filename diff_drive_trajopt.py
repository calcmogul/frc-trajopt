#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import List

from wpimath.geometry import Translation2d, Pose2d
from wpimath.system.plant import LinearSystemId
from wpimath.trajectory import DifferentialDriveTrajectoryOptimizer
from wpimath.trajectory.constraint import (
    BoxObstacleConstraint,
    CircleObstacleConstraint,
    DifferentialDriveMaxVelocityConstraint,
)


def plot_data(
    times: List[float],
    data: npt.NDArray[np.float64],
    labels: List[str],
    units: List[str],
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
        plt.title(f"{unit[:unit.find('(')].rstrip()} vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel(unit)
        for i in indices:
            if len(data.shape) > 1:
                d = data[i, :]
            else:
                d = data
            if len(indices) > 1:
                plt.plot(times, d, label=labels[i])
            else:
                plt.plot(times, d)
        if len(indices) > 1:
            plt.legend()


def curvature(xdot, ydot, xddot, yddot, v):
    return (xdot * yddot - ydot * xddot) / v**3


def main():
    trackwidth = 0.699  # m
    Kv_linear = 3.02  # V/(m/s)
    Ka_linear = 0.642  # V/(m/s²)
    Kv_angular = 1.382  # V/(m/s)
    Ka_angular = 0.08495  # V/(m/s²)

    system = LinearSystemId.identify_drivetrain_system(
        Kv_linear, Ka_linear, Kv_angular, Ka_angular
    )

    traj = DifferentialDriveTrajectoryOptimizer(
        system, trackwidth, 0.005, Pose2d(0, 0, 0)
    )
    traj.add_translation(
        Translation2d(4.5, 3), [DifferentialDriveMaxVelocityConstraint(2)]
    )
    traj.add_pose(Pose2d(4, 1, -math.pi))
    traj.add_constraint(CircleObstacleConstraint(2, 1.387, 1))
    traj.add_constraint(BoxObstacleConstraint(2, 1.387, 2, 2))
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

    # v = (v_l + v_r) / 2
    # x' = v cos(θ)
    # y' = v sin(θ)
    #
    # [a_l, a_r] = B⁻¹(dx/dt − Ax)
    # a = (a_l + a_r) / 2
    # x" = v' cos(θ) + v (-sin(θ) ω)
    #    = a cos(θ) - vω sin(θ)
    #    = a cos(θ) - v²/r sin(θ)
    # y" = v' sin(θ) + v (cos(θ) ω)
    #    = a sin(θ) + vω cos(θ)
    #    = a sin(θ) + v²/r cos(θ)
    #
    # κ = (x'y" - y'x") / √(x'² + y'²)³
    #   = (x'y" - y'x") / v³
    #   = ((v cos(θ))(a sin(θ) + ...?
    #
    # vk = w
    # k = w/v
    # k = (v_r - v_l) / trackwidth / ((v_l + v_r) / 2)
    #   = (v_r - v_l) / trackwidth * 2 / (v_l + v_r)
    #   = 2(v_r - v_l) / (trackwidth * (v_l + v_r))

    # Plot curvature vs time
    ks = []
    for k in range(states.shape[1]):
        ks.append(
            2
            * (states[4, k] - states[3, k])
            / (trackwidth * (states[3, k] + states[4, k]))
        )
    plt.figure()
    plt.title("Curvature vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Curvature (rad/m)")
    plt.plot(times, ks)

    dkdts = [0]
    for k in range(1, len(ks)):
        dkdts.append((ks[k] - ks[k - 1]) / (times[k] - times[k - 1]))
    plt.figure()
    plt.title("dk/dt vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("dk/dt (rad/m/s)")
    plt.plot(times, dkdts)

    d2kdt2s = [0]
    for k in range(1, len(ks)):
        d2kdt2s.append((dkdts[k] - dkdts[k - 1]) / (times[k] - times[k - 1]))
    plt.figure()
    plt.title("d²k/dt² vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("d²k/dt² (rad/m/s²)")
    plt.plot(times, d2kdt2s)

    plt.show()


if __name__ == "__main__":
    main()
