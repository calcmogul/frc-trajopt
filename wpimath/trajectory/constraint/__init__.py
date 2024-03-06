from abc import ABCMeta, abstractmethod

from jormungandr.autodiff import VariableMatrix
from wpimath.system import LinearSystem


class TrajectoryConstraint:
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(opti, X, U) -> None:
        pass


class DifferentialDriveCentripetalAccelerationConstraint(TrajectoryConstraint):
    def __init__(self, trackwidth: float, max_acceleration: float):
        self.trackwidth = trackwidth
        self.max_acceleration = max_acceleration

    def apply(self, opti, X, U) -> None:
        # a = v²/r
        # k = 1/r, so a = v²k
        # k = ω/v, so a = v²ω/v = vω
        # a = (v_l + v_r) / 2 * (v_r - v_l) / trackwidth
        # a = (v_r + v_l)(v_r - v_l) / (2 trackwidth)
        # a = (v_r² - v_l²) / (2 trackwidth)
        a = (X[4, :] ** 2 - X[3, :] ** 2) / (2 * self.trackwidth)
        opti.subject_to(-self.max_acceleration <= a)
        opti.subject_to(a <= self.max_acceleration)


class DifferentialDriveMaxVelocityConstraint(TrajectoryConstraint):
    def __init__(self, max_velocity: float):
        self.max_velocity = max_velocity

    def apply(self, opti, X, U) -> None:
        v = (X[3, :] + X[4, :]) / 2
        opti.subject_to(-self.max_velocity <= v)
        opti.subject_to(v <= self.max_velocity)


class DifferentialDriveMaxAccelerationConstraint(TrajectoryConstraint):
    def __init__(self, system: LinearSystem, max_acceleration: float):
        self.system = system
        self.max_acceleration = max_acceleration

    def apply(self, opti, X, U) -> None:
        dxdt = self.system.A @ X[3:5, :] + self.system.B @ U
        a = (dxdt[0, :] + dxdt[1, :]) / 2
        opti.subject_to(-self.max_acceleration <= a)
        opti.subject_to(a <= self.max_acceleration)


class BoxObstacleConstraint(TrajectoryConstraint):
    def __init__(self, center_x: float, center_y: float, width: float, height: float):
        self.center_x = center_x
        self.center_y = center_y
        self.r_x = width / 2.0
        self.r_y = height / 2.0

    def apply(self, opti, X, U) -> None:
        import math
        from jormungandr.autodiff import sqrt

        x = X[0, :]
        y = X[1, :]

        x_new = (x - VariableMatrix.ones(1, x.shape[1]) * self.center_x) / self.r_x
        y_new = (y - VariableMatrix.ones(1, y.shape[1]) * self.center_y) / self.r_y

        # |x| + |y| > 1
        #
        # x' = cos(θ) x + sin(θ) y
        # y' = -sin(θ) x + cos(θ) y
        #
        # Let θ=45°.
        #
        # x' = 1/√2 x + 1/√2 y
        # y' = -1/√2 x + 1/√2 y
        #
        # |1/√2 x + 1/√2 y| + |-1/√2 x + 1/√2 y| > 1
        # |1/√2(x + y)| + |1/√2(y - x)| > 1
        # 1/√2 |x + y| + 1/√2 |y - x| > 1
        # 1/√2 (|x + y| + |y - x|) > 1
        # |x + y| + |y - x| > √2
        # |x + y| + |y - x| > √2
        # √((x + y)²) + √((y − x)²) > √2
        for col in range(x_new.shape[1]):
            opti.subject_to(
                sqrt((x_new[0, col] + y_new[0, col]) ** 2)
                + sqrt((y_new[0, col] - x_new[0, col]) ** 2)
                > math.sqrt(2)
            )


class CircleObstacleConstraint(TrajectoryConstraint):
    def __init__(self, center_x: float, center_y: float, radius: float):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def apply(self, opti, X, U) -> None:
        for col in range(X.shape[1]):
            opti.subject_to(
                (X[0, col] - self.center_x) ** 2 + (X[1, col] - self.center_y) ** 2
                > self.radius**2
            )
