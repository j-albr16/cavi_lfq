"""Compare a fitted posterior with reference features (a featureXML or a simulation's truth)."""

import numpy as np

from .generative import DELTA_13C
from .priors import CHARGES, HyperParameters, VariationalParams


def assign_components(
    v: VariationalParams,
    p: HyperParameters,
    features: list[dict],
    num_j: int,
    min_ions: float = 1.0,
    tol_mz: float = 0.03,  # Da
    tol_rt: float = 1.0,  # min
) -> np.ndarray:
    """For every component the index of the reference feature it belongs to, or -1.

    A component belongs to a feature if its mass sits on one of the feature's isotope positions
    `m + j * Delta / c` (j < num_j, c the feature's charge) and its RT apex is within `tol_rt`.
    This groups together the fragments of one feature and its isotope teeth, which the fit may
    represent as separate components. If several features fit, the closest in mass wins.
    Components with at most `min_ions` expected ions are ignored.
    """
    n_k = np.asarray(v.alpha[1:]) - np.asarray(p.alpha)[1:]
    best_c = np.argmax(np.asarray(v.q_c), axis=1)
    owner = np.full(n_k.shape[0], -1)
    for k in np.flatnonzero(n_k > min_ions):
        mass, rt = float(v.m[k, best_c[k]]), float(v.mu[k])
        best = np.inf
        for f, feat in enumerate(features):
            if abs(rt - feat["rt_apex_min"]) >= tol_rt:
                continue
            charge = max(int(feat["charge"]), 1)
            for j in range(num_j):
                d = abs(mass - (feat["mono_mz"] + j * DELTA_13C / charge))
                if d < tol_mz and d < best:
                    owner[k], best = f, d
    return owner


def feature_intensities(
    v: VariationalParams,
    p: HyperParameters,
    features: list[dict],
    num_j: int,
    scale: float = 1.0,
    **kwargs,
) -> dict:
    """Intensity that the fit assigns to each reference feature, in the units of the file.

    It is `scale` times the expected ions of all components that belong to the feature (see
    `assign_components`), i.e. the summed intensity of the pixels given to them. `other` is the
    intensity of components that belong to no reference feature, `background` that of the
    uniform background component.
    """
    n_k = np.asarray(v.alpha[1:]) - np.asarray(p.alpha)[1:]
    owner = assign_components(v, p, features, num_j, **kwargs)
    F = len(features)
    return {
        "intensity": np.array([scale * n_k[owner == f].sum() for f in range(F)]),
        "n_components": np.array([int((owner == f).sum()) for f in range(F)]),
        "other": float(scale * n_k[owner < 0].sum()),
        "background": float(scale * (float(v.alpha[0]) - float(p.alpha[0]))),
        "owner": owner,
    }


def compare_intensities(measured: np.ndarray, reference: np.ndarray) -> dict:
    """Agreement of two intensity vectors: ratios and rank / log correlation over positive pairs."""
    measured, reference = np.asarray(measured, float), np.asarray(reference, float)
    ok = (measured > 0) & (reference > 0)
    ratio = np.where(ok, measured / np.where(reference > 0, reference, 1.0), np.nan)
    out = {"ratio": ratio, "n_compared": int(ok.sum()), "median_ratio": float(np.nanmedian(ratio)) if ok.any() else np.nan}
    if ok.sum() >= 3:
        lm, lr = np.log10(measured[ok]), np.log10(reference[ok])
        out["pearson_log"] = float(np.corrcoef(lm, lr)[0, 1])
        rank = lambda a: np.argsort(np.argsort(a)).astype(float)
        out["spearman"] = float(np.corrcoef(rank(measured[ok]), rank(reference[ok]))[0, 1])
        # spread of the ratio around its median, in factors
        out["ratio_spread"] = float(np.exp(np.std(np.log(ratio[ok]))))
    return out
