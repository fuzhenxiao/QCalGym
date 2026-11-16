from bisect import bisect_left, bisect_right
import numpy as np

def scalar_evaluator(solution):
    from scipy.integrate._ivp.rk import Dop853DenseOutput
    pieces = solution.interpolants
    if not solution.ascending or not pieces or any((type(p) is not Dop853DenseOutput or p.y_old.shape != (1,) or p.y_old.dtype not in (np.dtype('float64'), np.dtype('complex128')) for p in pieces)):
        return lambda t: solution(t)[0]
    times = solution.ts_sorted.tolist()
    search = bisect_left if solution.side == 'left' else bisect_right
    coefficients = [(float(p.t_old), float(p.h), p.y_old[0].item(), tuple((f[0].item() for f in reversed(p.F)))) for p in pieces]
    last = len(coefficients) - 1
    scalar_type = pieces[0].y_old.dtype.type

    def evaluate(t):
        t = float(t)
        index = min(max(search(times, t) - 1, 0), last)
        start, step, initial, terms = coefficients[index]
        x = (t - start) / step
        other = 1 - x
        y = 0j if isinstance(initial, complex) else 0.0
        for i, term in enumerate(terms):
            y += term
            y *= x if i % 2 == 0 else other
        return scalar_type(y + initial)
    return evaluate
