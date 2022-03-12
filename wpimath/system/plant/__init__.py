import numpy as np

from wpimath.system import LinearSystem


class LinearSystemId:
    @staticmethod
    def identify_drivetrain_system(
        Kv_linear, Ka_linear, Kv_angular, Ka_angular
    ) -> LinearSystem:
        """Constructs the state-space model for a 2 DOF drivetrain velocity
        system from system identification data.

        States: [[left velocity], [right velocity]]
        Inputs: [[left voltage], [right voltage]]
        Outputs: [[left velocity], [right velocity]]

        Throws RuntimeError if kv_linear <= 0, ka_linear <= 0, kv_angular <= 0,
        or ka_angular <= 0.

        Keyword arguments:
        Kv_linear -- The linear velocity gain in volts per (meter per second).
        Ka_linear -- The linear acceleration gain in volts per (meter per
                     second squared).
        Kv_angular -- The angular velocity gain in volts per (meter per second).
        Ka_angular -- The angular acceleration gain in volts per (meter per
                      second squared).
        """
        if Kv_linear <= 0:
            raise RuntimeError("Kv,linear must be greater than zero.")
        if Ka_linear <= 0:
            raise RuntimeError("Ka,linear must be greater than zero.")
        if Kv_angular <= 0:
            raise RuntimeError("Kv,angular must be greater than zero.")
        if Ka_angular <= 0:
            raise RuntimeError("Ka,angular must be greater than zero.")

        A1 = -(Kv_linear / Ka_linear + Kv_angular / Ka_angular)
        A2 = -(Kv_linear / Ka_linear - Kv_angular / Ka_angular)
        B1 = 1 / Ka_linear + 1 / Ka_angular
        B2 = 1 / Ka_linear - 1 / Ka_angular

        A = 0.5 * np.array([[A1, A2], [A2, A1]])
        B = 0.5 * np.array([[B1, B2], [B2, B1]])
        C = np.eye(2)
        D = np.zeros((2, 2))

        return LinearSystem(A, B, C, D)
