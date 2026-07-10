"""One Euro Filter (Casiez, Roussel, Vogel 2012).

Adaptive low-pass filter: heavy smoothing at low velocity, light smoothing
at high velocity. Replaces fixed-alpha EMA for per-landmark coordinate
smoothing so that fast transients (e.g. swim catch, hand recovery) are not
blurred while idle jitter is still suppressed.

NaN-safe: when the input sample is NaN the filter returns the last valid
estimate (or NaN if no sample has been seen yet). This lets the upstream
visibility gate mark low-confidence landmarks as NaN and have gaps
propagated cleanly instead of crashing.
"""

from __future__ import annotations

import math


class OneEuro:
    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, freq: float | None = None) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return self._x_prev if self._x_prev is not None else float("nan")

        f = freq or self.freq
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        dx = (x - self._x_prev) * f
        a_d = self._alpha(self.d_cutoff, f)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, f)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat
