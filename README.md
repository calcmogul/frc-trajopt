# FRC Trajectory Optimization

This repository uses [CasADi](https://web.casadi.org/) to formulate trajectory
planning as an optimization problem.

## Features

* Differential drive dynamics constraint
* Arbitrarily ordered list of translation and pose waypoints
* Per-waypoint constraints
  * Max velocity
  * Max acceleration
  * Max centripetal acceleration
* Automatically determines whether the robot should reverse directions to reach
  the destination the fastest
  * The solver doesn't do anything special for this; it's just a side effect of
    the problem formulation.

## Usage

In `diff_drive_trajopt.py`, the user provides
DifferentialDriveTrajectoryOptimizer with a differential drive model, a mix of
translations and waypoints, and optional per-segment constraints. The solver
gives a time optimal trajectory satisfying all those constraints after 3 to 4
seconds.

## Future work

We're still working on ways to speed up the solver, but real-time trajectory
optimization is still a ways off. The main thing slowing it down is the number
of decision variables it's optimizing (~1400).

If the number of samples is reduced, the solver can finish in around 600 ms, but
it will occasionally not find a solution. There's methods for approximating the
dynamics with splines to reduce the number of samples that show promise (e.g.,
direct collocation), but they produce strange artifacts like kinks and
loopty-loops that are less optimal than the original solution.

The initial guess for the solver could also be improved. It's currently just
linear interpolation between the waypoints with the velocities initialized to
zero. If we generate something smoother with splines, the solver will converge
faster. For example, using a linear time-varying LQR cut the solve time in half.
However, running the LQR in the first place takes about the same amount of time
it saved in the solver.

This general toolset can also support obstacle avoidance, if there's a demand
for that. It can be computationally intensive though.

Swerve drive support is planned. Since the user specifies translations and
headings rather than the trajectory shape, the only thing that needs to be done
to support swerve is swapping out the dynamics constraint.
