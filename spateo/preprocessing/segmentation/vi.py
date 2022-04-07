"""Variational inference implementation of a negative binomial mixture model
using Pyro.
"""
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.nn import PyroModule, PyroParam
from pyro.optim import Adam
from tqdm import tqdm

from ...errors import PreprocessingError


class NegativeBinomialMixture(PyroModule):
    def __init__(self, x: np.ndarray, n: int = 2):
        super().__init__()
        self.x = torch.tensor(x.astype(np.float32))
        self.n = n
        self.w = PyroParam(torch.randn(n))
        self.counts = PyroParam(torch.randn(n))
        self.logits = PyroParam(torch.randn(n))
        self.__optimizer = None

    def optimizer(self):
        if self.__optimizer is None:
            self.__optimizer = Adam({"lr": 0.1})
        return self.__optimizer

    def get_params(self, train=False):
        w = self.w
        counts = F.softplus(self.counts)
        logits = self.logits
        if not train:
            w = w.detach().numpy()
            counts = counts.detach().numpy()
            logits = logits.detach().numpy()
        return dict(w=w, counts=counts, logits=logits)

    def forward(self, x):
        params = self.get_params(train=True)
        w, counts, logits = params["w"], params["counts"], params["logits"]

        with pyro.plate("x", size=len(x)):
            assignment = pyro.sample("assignment", dist.Categorical(logits=w), infer={"enumerate": "parallel"})
            pyro.sample(
                "obs", dist.NegativeBinomial(counts[assignment], logits=logits[assignment], validate_args=False), obs=x
            )

    def train(self, n_epochs: int = 1000):
        optimizer = self.optimizer()
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        guide = AutoDelta(poutine.block(self, expose=["w", "counts", "logits"]))
        svi = SVI(self, guide, optimizer, elbo)

        with tqdm(total=n_epochs) as pbar:
            for _ in range(n_epochs):
                loss = svi.step(self.x) / self.x.numel()

                pbar.set_description(f"Loss {loss:.4e}")
                pbar.update(1)

    @staticmethod
    def conditionals(params, x):
        counts, logits = params["counts"], params["logits"]
        x = torch.tensor(x.astype(np.float32))
        n = len(counts)
        dists = [dist.NegativeBinomial(c, logits=l) for c, l in zip(counts, logits)]
        return tuple(torch.exp(d.log_prob(x)).numpy() for d in sorted(dists, key=lambda d: d.mean))


def conditionals(
    X: np.ndarray,
    vi_results: Union[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    ],
    bins: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the conditional probabilities, for each pixel, of observing the
    observed number of UMIs given that the pixel is background/foreground.

    Args:
        X: UMI counts per pixel
        em_results: Return value of :func:`run_em`.
        bins: Pixel bins, as was passed to :func:`run_em`.

    Returns:
        Two Numpy arrays, the first corresponding to the background conditional
        probabilities, and the second to the foreground conditional probabilities

    Raises:
        PreprocessingError: If `em_results` is a dictionary but `bins` was not
            provided.
    """
    if isinstance(vi_results, dict):
        if bins is None:
            raise PreprocessingError("`em_results` indicate binning was used, but `bins` was not provided")
        background_cond = np.ones(X.shape)
        cell_cond = np.zeros(X.shape)
        for label, params in vi_results.items():
            mask = bins == label
            background_cond[mask], cell_cond[mask] = NegativeBinomialMixture.conditionals(
                {"counts": params[0], "logits": params[-1]}, X[mask]
            )
    else:
        params = vi_results
        background_cond, cell_cond = NegativeBinomialMixture.conditionals(
            {"counts": params[0], "logits": params[-1]}, X
        )

    return background_cond, cell_cond


def run_vi(
    X: np.ndarray,
    downsample: Union[int, float] = 1.0,
    n_epochs: int = 1000,
    bins: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Union[
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
]:
    samples = {}  # key 0 when bins = None
    if bins is not None:
        for label in np.unique(bins):
            if label > 0:
                samples[label] = X[bins == label]
    else:
        samples[0] = X.flatten()

    downsample_scale = True
    if downsample > 1:
        downsample_scale = False
    rng = np.random.default_rng(seed)
    final_samples = {}
    total = sum(len(_samples) for _samples in samples.values())
    for label, _samples in samples.items():
        _downsample = int(len(_samples) * downsample) if downsample_scale else int(downsample * (len(_samples) / total))
        if len(_samples) > _downsample:
            weights = np.log1p(_samples + 1)
            _samples = rng.choice(_samples, _downsample, replace=False, p=weights / weights.sum())
        final_samples[label] = np.array(_samples)

    results = {}
    for label, _samples in final_samples.items():
        pyro.clear_param_store()
        nbm = NegativeBinomialMixture(_samples)
        nbm.train(n_epochs)
        params = nbm.get_params()
        results[label] = params["counts"], params["logits"]

    return results if bins is not None else results[0]
