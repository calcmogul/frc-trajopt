from abc import ABCMeta, abstractmethod

from wpimath.system import LinearSystem


class TrajectoryConstraint:
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(opti, X, U) -> None:
        pass


class CentripetalAccelerationConstraint(TrajectoryConstraint):
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
        opti.subject_to(
            opti.bounded(
                -self.max_acceleration,
                (X[4, :] ** 2 - X[3, :] ** 2) / (2 * self.trackwidth),
                self.max_acceleration,
            )
        )


class MaxVelocityConstraint(TrajectoryConstraint):
    def __init__(self, max_velocity: float):
        self.max_velocity = max_velocity

    def apply(self, opti, X, U) -> None:
        v = (X[3, :] + X[4, :]) / 2
        opti.subject_to(opti.bounded(-self.max_velocity, v, self.max_velocity))


class MaxAccelerationConstraint(TrajectoryConstraint):
    def __init__(self, system: LinearSystem, max_acceleration: float):
        self.system = system
        self.max_acceleration = max_acceleration

    def apply(self, opti, X, U) -> None:
        dxdt = self.system.A @ X[3:5, :] + self.system.B @ U
        a = (dxdt[0, :] + dxdt[1, :]) / 2
        opti.subject_to(opti.bounded(-self.max_acceleration, a, self.max_acceleration))
