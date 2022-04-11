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
    def __init__(self, x: np.ndarray, n: int = 2, n_init: int = 5, seed: Optional[int] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.x = torch.tensor(x.astype(np.float32))
        self.n = n
        self.scale = torch.median(self.x[self.x > 0])

        params = self.init_best_params(n_init)
        self.w = PyroParam(params["w"])
        self.counts = PyroParam(params["counts"])
        self.logits = PyroParam(params["logits"])
        self.__optimizer = None

    def init_best_params(self, n_init):
        best_log_prob = -np.inf
        best_params = None
        for _ in range(n_init):
            self.w = torch.randn(self.n)
            self.counts = torch.randn(self.n)
            self.logits = torch.randn(self.n)
            params = self.get_params(True)
            w, counts, logits = params["w"], params["counts"], params["logits"]

            assignment = dist.Categorical(logits=w).sample(self.x.size())
            log_prob = dist.NegativeBinomial(
                counts[assignment], logits=logits[assignment], validate_args=False
            ).log_prob(self.x)
            if log_prob.sum() > best_log_prob:
                best_log_prob = log_prob.sum()
                best_params = self.get_params(True, False)
        return best_params

    def optimizer(self):
        if self.__optimizer is None:
            self.__optimizer = Adam({"lr": 0.1})
        return self.__optimizer

    def get_params(self, train=False, transform=True):
        w, counts, logits = self.w, self.counts, self.logits

        if transform:
            counts = F.softplus(self.counts) * self.scale
        if not train:
            w = w.detach().numpy()
            counts = counts.detach().numpy()
            logits = logits.detach().numpy()
        return dict(w=w, counts=counts, logits=logits)

    def forward(self, x):
        params = self.get_params(True)
        w, counts, logits = params["w"], params["counts"], params["logits"]

        with pyro.plate("x", size=len(x)):
            assignment = pyro.sample("assignment", dist.Categorical(logits=w), infer={"enumerate": "parallel"})
            pyro.sample(
                "obs", dist.NegativeBinomial(counts[assignment], logits=logits[assignment], validate_args=False), obs=x
            )

    def train(self, n_epochs: int = 500):
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
            conditionals = NegativeBinomialMixture.conditionals({"counts": params[0], "logits": params[1]}, X[mask])
            background_cond[mask], cell_cond[mask] = conditionals[0], conditionals[-1]
    else:
        params = vi_results
        conditionals = NegativeBinomialMixture.conditionals({"counts": params[0], "logits": params[1]}, X)
        background_cond, cell_cond = conditionals[0], conditionals[-1]

    return background_cond, cell_cond


def run_vi(
    X: np.ndarray,
    downsample: Union[int, float] = 0.01,
    n_epochs: int = 500,
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
            _samples = rng.choice(_samples, _downsample, replace=False)
        final_samples[label] = np.array(_samples)

    results = {}
    for label, _samples in final_samples.items():
        pyro.clear_param_store()
        nbm = NegativeBinomialMixture(_samples, seed=seed)
        nbm.train(n_epochs)
        params = nbm.get_params()
        results[label] = params["counts"], params["logits"]

    return results if bins is not None else results[0]
