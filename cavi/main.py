import jax 
import jax.numpy as jnp
from jaxtyping import Array, Float
import openms as oms

from .priors import HyperParameters, VariationalParams
from .z_j import get_r
from .variational_updates import pi, nig, m_k, q_c, peak_width, carbon_count, heavy_isotope
from .elbo import elbo

def get_gamma() -> float:
    return 1.

def load_spectrum(path: str) -> tuple[Float[Array, "N"],Float[Array, "N"],Float[Array, "N"]]:
    exp = oms.MSExperiment()
    oms.MzMLFile().load("run.mzML", exp)
    t, y, w = [], [], []
    for spec in exp:
        if spec.getMSLevel() != 1: continue           # only MS1 survey scans
        mz, inten = spec.get_peaks()
        t.append(jnp.full(mz.shape, spec.getRT())) 
        y.append(mz)
        w.append(inten)
    t, y, w = map(jnp.concatenate, (t, y, w))
    w = get_gamma() * w                                     # intensities -> effective ion counts
    return t, y, w

def run(
    spectrum: Float[Array, "R M"], # spectrum with R retention time and M m/z
    priors: HyperParameters,
):

