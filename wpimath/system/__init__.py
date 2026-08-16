import numpy as np


class LinearSystem:
    """A plant defined using state-space notation.

    A plant is a mathematical model of a system's dynamics.

    For more on the underlying math, read
    https://file.tavsys.net/control/controls-engineering-in-frc.pdf.
    """

    def __init__(self, A, B, C, D):
        """Constructs a discrete plant with the given continuous system
        coefficients.

        Throws RuntimeError if any matrix element isn't finite.

        Keyword arguments:
        A -- System matrix.
        B -- Input matrix.
        C -- Output matrix.
        D -- Feedthrough matrix.
        """
        if not np.isfinite(A).all():
            raise RuntimeError(
                "Elements of A aren't finite. This is usually due to model "
                "implementation errors."
            )
        if not np.isfinite(B).all():
            raise RuntimeError(
                "Elements of B aren't finite. This is usually due to model "
                "implementation errors."
            )
        if not np.isfinite(C).all():
            raise RuntimeError(
                "Elements of C aren't finite. This is usually due to model "
                "implementation errors."
            )
        if not np.isfinite(D).all():
            raise RuntimeError(
                "Elements of D aren't finite. This is usually due to model "
                "implementation errors."
            )

        self.A = A
        self.B = B
        self.C = C
        self.D = D


class Models:
    @staticmethod
    def differential_drive_from_sysid(
        Kv_linear, Ka_linear, Kv_angular, Ka_angular
    ) -> LinearSystem:
        """
        Constructs the state-space model for a 2 DOF drivetrain velocity system
        from system identification data.

        States: [[left velocity], [right velocity]]
        Inputs: [[left voltage], [right voltage]]
        Outputs: [[left velocity], [right velocity]]

        Args:
            Kv_linear: The linear velocity gain in V/(m/s).
            Ka_linear: The linear acceleration gain in V/(m/s²).
            Kv_angular: The angular velocity gain in V/(m/s).
            Ka_angular: The angular acceleration gain in V/(m/s²).

        Raises: RuntimeError: If kv_linear ≤ 0, ka_linear ≤ 0, kv_angular ≤ 0,
            or ka_angular ≤ 0.

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
