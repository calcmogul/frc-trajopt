#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import List

from wpimath.geometry import Translation2d, Pose2d
from wpimath.system.plant import LinearSystemId
from wpimath.trajectory import DifferentialDriveTrajectoryOptimizer
from wpimath.trajectory.constraint import MaxVelocityConstraint


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
    traj.add_translation(Translation2d(4.5, 3), [MaxVelocityConstraint(2)])
    traj.add_pose(Pose2d(4, 1, -math.pi))
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
