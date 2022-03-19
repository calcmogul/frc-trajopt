import math

import casadi as ca
import numpy as np
import numpy.typing as npt
from typing import List, Optional
from wpimath.geometry import Translation2d, Pose2d
from wpimath.math_util import lerp
from wpimath.system import LinearSystem
from wpimath.trajectory.constraint import *


def runge_kutta(f, x, u, dt):
    """Fourth order Runge-Kutta integration.

    Keyword arguments:
    f -- vector function to integrate
    x -- vector of states
    u -- vector of inputs (constant for dt)
    dt -- time for which to integrate
    """
    half_dt = dt * 0.5
    k1 = f(x, u)
    k2 = f(x + half_dt * k1, u)
    k3 = f(x + half_dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class DifferentialDriveTrajectoryOptimizer:
    class Waypoint:
        def __init__(
            self,
            x: float,
            y: float,
            heading: Optional[float] = None,
            constraints: List[TrajectoryConstraint] = None,
        ) -> None:
            """Constructs a waypoint as either a pose or translation with an
            optional list of constraints to apply between this waypoint and the
            previous one.

            Keyword arguments:
            x -- the waypoint x position
            y -- the waypoint y position
            heading -- the waypoint heading (optional)
            constraints -- the list of constraints to apply (optional)
            """
            self.x = x
            self.y = y
            self.heading = heading
            if constraints is not None:
                self.constraints = constraints
            else:
                self.constraints = []

    def __init__(
        self, system: LinearSystem, trackwidth: float, dt: float, initial_pose: Pose2d
    ) -> None:
        """Constructs a differential drive trajectory optimizer.

        Keyword arguments:
        system -- the differential drive's velocity dynamics
        trackwidth -- the differential drive's trackwidth
        dt -- the sample period
        initial_pose -- the differential drive's initial pose
        """
        self.A = system.A
        self.B = system.B
        self.trackwidth = trackwidth
        self.dt = dt

        self.problem = ca.Opti()
        self.waypoints = [
            DifferentialDriveTrajectoryOptimizer.Waypoint(
                initial_pose.x, initial_pose.y, initial_pose.rotation
            )
        ]
        self.constraints = []  # List of TrajectoryConstraints

        self.vars_per_segment = 100

    def add_pose(
        self, pose: Pose2d, constraints: List[TrajectoryConstraint] = None
    ) -> None:
        """Add a new trajectory segment terminated by the given pose with an
        optional list of constraints to apply within that segment.

        Keyword arguments:
        pose -- the pose to add
        constraints -- the list of constraints to apply (optional)
        """
        self.waypoints.append(
            DifferentialDriveTrajectoryOptimizer.Waypoint(
                pose.x, pose.y, pose.rotation, constraints
            )
        )

    def add_translation(
        self, translation: Translation2d, constraints: List[TrajectoryConstraint] = None
    ) -> None:
        """Add a new trajectory segment terminated by the given translation
        (i.e., no heading constraint) with an optional list of constraints to
        apply within that segment.

        Keyword arguments:
        translation -- the translation to add
        constraints -- the list of constraints to apply (optional)
        """
        self.waypoints.append(
            DifferentialDriveTrajectoryOptimizer.Waypoint(
                translation.x, translation.y, constraints=constraints
            )
        )

    def add_constraint(self, constraint: TrajectoryConstraint) -> None:
        """Add the given constraint to all trajectory segments.

        Keyword arguments:
        constraint -- the constraint to apply
        """
        self.constraints.append(constraint)

    def f_diff_drive(
        self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """The dynamical model for a differential drive.

        Keyword arguments:
        x -- state column vector: [x, y, heading, left velocity, right velocity]
        u -- input column vector: [left voltage, right voltage]

        Returns:
        dx/dt
        """
        v = (x[3, 0] + x[4, 0]) / 2

        dxdt = np.empty((5, 1))
        dxdt[0, 0] = v * math.cos(x[2, 0])
        dxdt[1, 0] = v * math.sin(x[2, 0])
        dxdt[2, 0] = (x[4, 0] - x[3, 0]) / self.trackwidth
        dxdt[3:5, :] = self.A @ x[3:5, :] + self.B @ u
        return dxdt

    def generate_initial_path(self, X, U):
        """Use differential drive LQR to generate the initial path for
        trajectory optimization.
        """
        import control as ct
        from frccontrol import lqr

        A = np.array(
            [
                [0, 0, 0, 0.5, 0.5],
                [0, 0, 0, 0, 0],
                [0, 0, 0, -1 / self.trackwidth, 1 / self.trackwidth],
                [0, 0, 0, self.A[0, 0], self.A[0, 1]],
                [0, 0, 0, self.A[1, 0], self.A[1, 1]],
            ]
        )
        B = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [self.B[0, 0], self.B[0, 1]],
                [self.B[1, 0], self.B[1, 1]],
            ]
        )
        Q = np.diag(1.0 / np.square([0.0625, 0.125, 2.5, 0.95, 0.95]))
        R = np.diag(1.0 / np.square([12, 12]))

        i_waypoint = 0
        waypoint = self.waypoints[i_waypoint]
        x = np.array([[waypoint.x], [waypoint.y], [0], [0], [0]])
        if waypoint.heading is not None:
            x[2, 0] = waypoint.heading

        i_waypoint += 1
        waypoint = self.waypoints[i_waypoint]
        r = np.array([[waypoint.x], [waypoint.y], [0], [0], [0]])
        if waypoint.heading is not None:
            r[2, 0] = waypoint.heading
        u = np.array([[0], [0]])
        for k in range(X.shape[1] - 1):
            self.problem.set_initial(X[:, k], x)

            v = (x[3, 0] + x[4, 0]) / 2
            if abs(v) < 1e-8:
                v = 1e-8
            A[1, 2] = v
            sys = ct.StateSpace(A, B, np.eye(5, 5), np.zeros((5, 2))).sample(self.dt)
            K = lqr(sys, Q, R)
            rot = np.array(
                [
                    [math.cos(x[2]), math.sin(x[2]), 0, 0, 0],
                    [-math.sin(x[2]), math.cos(x[2]), 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
            u = K @ rot @ (r - x)
            u = np.clip(u, np.array([[-12], [-12]]), np.array([[12], [12]]))
            self.problem.set_initial(U[:, k], u)
            x = runge_kutta(self.f_diff_drive, x, u, self.dt)

            if np.allclose(r, x):
                i_waypoint += 1
                waypoint = self.waypoints[i_waypoint]
                r = np.array([[waypoint.x], [waypoint.y], [0], [0], [0]])
                if waypoint.heading is not None:
                    r[2, 0] = waypoint.heading

    def optimize(self, q: float, r: List[float]):
        """Generate the optimal trajectory.

        Keyword arguments:
        q -- minimum time cost weight
        r -- list of the maximum allowed excursions of the control inputs from
             no actuation

        Returns:
        times -- list of times in solution
        states -- matrix of states (states x times)
        inputs -- matrix of inputs (inputs x times)
        """
        num_segments = len(self.waypoints) - 1
        num_vars = self.vars_per_segment * num_segments

        # States: [x, y, heading, left velocity, right velocity]
        X = self.problem.variable(5, num_vars + 1)

        # Inputs: [left voltage, right voltage]
        U = self.problem.variable(2, num_vars)

        # Apply waypoint constraints
        for i, waypoint in enumerate(self.waypoints):
            # If it's the first waypoint, constrain the first state. Otherwise,
            # constrain the last one.
            if i == 0:
                segment_end = 0
            else:
                segment_end = i * self.vars_per_segment - 1

            self.problem.subject_to(X[0, segment_end] == waypoint.x)
            self.problem.subject_to(X[1, segment_end] == waypoint.y)
            if waypoint.heading is not None:
                self.problem.subject_to(
                    ca.cos(X[2, segment_end]) == math.cos(waypoint.heading)
                )
                self.problem.subject_to(
                    ca.sin(X[2, segment_end]) == math.sin(waypoint.heading)
                )
            if i > 0:
                for constraint in waypoint.constraints:
                    segment_start = (i - 1) * self.vars_per_segment
                    constraint.apply(
                        self.problem,
                        X[:, segment_start:segment_end],
                        U[:, segment_start:segment_end],
                    )

        self.generate_initial_path(X, U)

        # Set up duration decision variables
        Ts = []
        dts = []
        for segment in range(num_segments):
            T = self.problem.variable()
            self.problem.subject_to(T >= 0)
            self.problem.set_initial(T, 1)
            Ts.append(T)

            dt = T / self.vars_per_segment
            dts.append(dt)

        # Linear cost on time
        J = q * sum(Ts)

        # Quadratic cost on control input
        for k in range(num_vars):
            R = np.diag(1 / np.square(r))
            J += U[:, k].T @ R @ U[:, k] * dts[int(k / self.vars_per_segment)]

        self.problem.minimize(J)

        # Dynamics constraint
        for k in range(num_vars):
            # RK4 integration
            h = dts[int(k / self.vars_per_segment)]
            x_k = X[:, k]
            u_k = U[:, k]
            x_k1 = X[:, k + 1]

            k1 = self.f(x_k, u_k)
            k2 = self.f(x_k + h / 2 * k1, u_k)
            k3 = self.f(x_k + h / 2 * k2, u_k)
            k4 = self.f(x_k + h * k3, u_k)
            self.problem.subject_to(x_k1 == x_k + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

        # Constrain start and end velocities to zero
        self.problem.subject_to(X[3:5, 0] == np.zeros((2, 1)))
        self.problem.subject_to(X[3:5, -1] == np.zeros((2, 1)))

        # Input constraint
        self.problem.subject_to(self.problem.bounded(-r[0], U[0, :], r[0]))
        self.problem.subject_to(self.problem.bounded(-r[1], U[1, :], r[1]))

        # Apply custom constraints
        for constraint in self.constraints:
            constraint.apply(self.problem, X, U)

        self.problem.solver("ipopt")
        sol = self.problem.solve()

        # Generate times for time domain plots
        times = [0]
        for k in range(num_vars):
            times.append(times[-1] + sol.value(dts[int(k / self.vars_per_segment)]))

        # Resample trajectory at 5 ms period
        return DifferentialDriveTrajectoryOptimizer.resample(
            times, sol.value(X), sol.value(U), self.dt
        )

    def f(
        self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """The dynamical model for a differential drive.

        Keyword arguments:
        x -- state column vector: [x, y, heading, left velocity, right velocity]
        u -- input column vector: [left voltage, right voltage]

        Returns:
        dx/dt
        """
        v = (x[3] + x[4]) / 2

        return ca.vertcat(
            v * ca.cos(x[2]),
            v * ca.sin(x[2]),
            (x[4] - x[3]) / self.trackwidth,
            self.A @ x[3:5] + self.B @ u,
        )

    @staticmethod
    def resample(times, states, inputs, dt):
        """Resample the given states and inputs at a new period.

        Keyword arguments:
        times -- list of times in solution
        states -- matrix of states (states x times)
        inputs -- matrix of inputs (inputs x times)
        """
        new_times = [times[0]]
        new_states = states[:, 0:1]
        new_inputs = inputs[:, 0:1]

        # Start at 1 because we use the previous sample later on for
        # interpolation
        sample = 1

        while sample < len(times) - 1 and new_times[-1] < times[-1]:
            # Find first sample >= the requested timestamp
            while sample < len(times) - 1 and times[sample] < new_times[-1] + dt:
                sample += 1

            if sample == len(times) - 1:
                break

            prev_sample = sample - 1

            if abs(times[sample] - times[prev_sample]) < 1e-9:
                # If the difference in sample times is negligible, use that
                # sample
                new_states = np.concatenate(
                    (new_states, states[:, sample : sample + 1]), axis=1
                )
                new_inputs = np.concatenate(
                    (new_inputs, inputs[:, sample : sample + 1]), axis=1
                )
            else:
                # Interpolate between previous and current sample
                x = lerp(
                    states[:, prev_sample : prev_sample + 1],
                    states[:, sample : sample + 1],
                    ((new_times[-1] + dt) - times[prev_sample])
                    / (times[sample] - times[prev_sample]),
                )
                new_states = np.concatenate((new_states, x), axis=1)
                u = lerp(
                    inputs[:, prev_sample : prev_sample + 1],
                    inputs[:, sample : sample + 1],
                    ((new_times[-1] + dt) - times[prev_sample])
                    / (times[sample] - times[prev_sample]),
                )
                new_inputs = np.concatenate((new_inputs, u), axis=1)
            new_times.append(new_times[-1] + dt)

        # Add last sample as final element
        new_times.append(new_times[-1] + dt)
        new_states = np.concatenate((new_states, states[:, -1:]), axis=1)

        return new_times, new_states, new_inputs
