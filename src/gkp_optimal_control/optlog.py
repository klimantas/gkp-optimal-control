"""Per-seed optimizer diagnostics for the sweep scripts.

Every sweep in this project is a multi-start L-BFGS-B: several seeds are run, the
best is kept, and the rest are discarded. That discard is what makes three claims
in the write-up uncheckable after the fact --- that a particular row is "a stalled
line search rather than a converged point" (an L-BFGS status code we threw away),
that scatter is "about +/-0.05 in D_path" (a spread over seeds we never recorded),
and that an anomaly is "more likely an optimization failure than a real effect"
(a conjecture that per-seed outcomes would settle).

:func:`record` appends one row per L-BFGS call. The cost is a few hundred floats
per sweep against the kilobytes of parameters already saved, so it is always on.

The stored ``status`` is scipy's L-BFGS-B code: 0 converged, 1 hit ``maxiter``,
2 abnormal termination (the line search failed to make progress) -- the last is
exactly the "stalled" case, so it stops being a judgement call.
"""

import numpy as np

COLUMNS = ("knob", "fun", "nit", "nfev", "status", "gnorm", "F", "D_path", "R_length")


def record(rows, tags, knob, seed_tag, res, metrics) -> None:
    """Append one L-BFGS outcome to ``rows`` (numeric) and ``tags`` (seed labels).

    Parameters
    ----------
    rows, tags : list
        Accumulators; ``rows`` takes a tuple matching :data:`COLUMNS`, ``tags``
        the seed label (``"cold"``, ``"warm"``, a perturbation radius, ...).
    knob : float
        The sweep variable for this row -- ``lambda`` or the waypoint count.
    seed_tag : str
        Where this start came from.
    res : scipy.optimize.OptimizeResult
        The finished L-BFGS-B result.
    metrics : dict
        Geometry of *this seed's* solution, not the winner's.
    """
    jac = np.asarray(res.jac) if getattr(res, "jac", None) is not None else np.array([np.nan])
    rows.append((
        float(knob), float(res.fun), int(res.nit), int(res.nfev), int(res.status),
        float(np.max(np.abs(jac))),
        float(metrics["F"]), float(metrics["D_path"]), float(metrics["R_length"]),
    ))
    tags.append(str(seed_tag))


def summarize(rows, tags) -> str:
    """One line per sweep point: winner, spread across seeds, and any bad status."""
    a = np.asarray(rows, dtype=float)
    if a.size == 0:
        return "(no optimizer diagnostics recorded)"
    out = [f"{'knob':>7} {'seeds':>6} {'best F':>8} {'best D':>8} "
           f"{'D spread':>9} {'max nit':>8} {'status':>12}"]
    for k in np.unique(a[:, 0]):
        g = a[a[:, 0] == k]
        win = g[np.argmin(g[:, 1])]                      # smallest objective
        bad = [f"{int(s)}x{int((g[:, 4] == s).sum())}" for s in np.unique(g[:, 4]) if s != 0]
        out.append(f"{k:7.3g} {len(g):6d} {win[6]:8.4f} {win[7]:8.4f} "
                   f"{g[:, 7].max() - g[:, 7].min():9.4f} {int(g[:, 2].max()):8d} "
                   f"{(','.join(bad) if bad else 'all converged'):>12}")
    return "\n".join(out)
