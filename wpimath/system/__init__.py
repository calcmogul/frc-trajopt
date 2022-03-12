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
