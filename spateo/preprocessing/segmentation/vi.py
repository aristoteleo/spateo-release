"""Variational inference implementation of a negative binomial mixture model
using Pyro.
"""
import itertools
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
    def __init__(
        self, x: np.ndarray, n: int = 2, n_init: int = 5, zero_inflated: bool = False, seed: Optional[int] = None
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.zero_inflated = zero_inflated
        self.x = torch.tensor(x.astype(np.float32))
        self.n = n
        self.scale = torch.median(self.x[self.x > 0])

        self.init_best_params(n_init)
        self.__optimizer = None

    def assignment(self, train=False):
        params = self.get_params(train)
        w = params["w"]
        return dist.Categorical(logits=w)

    def dist(self, assignment, train=False):
        params = self.get_params(train)
        counts, logits = params["counts"], params["logits"]
        if self.zero_inflated:
            z = params["z"]
            return dist.ZeroInflatedNegativeBinomial(
                counts[assignment], logits=logits[assignment], gate_logits=z[assignment], validate_args=False
            )
        return dist.NegativeBinomial(counts[assignment], logits=logits[assignment], validate_args=False)

    def init_best_params(self, n_init):
        best_log_prob = -np.inf
        best_params = None
        for _ in range(n_init):
            if self.zero_inflated:
                self.z = torch.randn(self.n)
            self.w = torch.randn(self.n)
            self.counts = torch.randn(self.n)
            self.logits = torch.randn(self.n)
            assignment = self.assignment(True).sample(self.x.size())
            log_prob = self.dist(assignment, True).log_prob(self.x)
            if log_prob.sum() > best_log_prob:
                best_log_prob = log_prob.sum()
                best_params = self.get_params(True, False)
        if self.zero_inflated:
            self.z = PyroParam(best_params["z"])
        self.w = PyroParam(best_params["w"])
        self.counts = PyroParam(best_params["counts"])
        self.logits = PyroParam(best_params["logits"])

    def optimizer(self):
        if self.__optimizer is None:
            self.__optimizer = Adam({"lr": 0.1})
        return self.__optimizer

    def get_params(self, train=False, transform=True):
        w, counts, logits = self.w, self.counts, self.logits
        if self.zero_inflated:
            z = self.z

        if transform:
            counts = F.softplus(self.counts) * self.scale
        if not train:
            if self.zero_inflated:
                z = z.detach().numpy()
            w = w.detach().numpy()
            counts = counts.detach().numpy()
            logits = logits.detach().numpy()
        params = dict(w=w, counts=counts, logits=logits)
        if self.zero_inflated:
            params["z"] = z
        return params

    def forward(self, x):
        with pyro.plate("x", size=len(x)):
            assignment = pyro.sample("assignment", self.assignment(True), infer={"enumerate": "parallel"})
            pyro.sample("obs", self.dist(assignment, True), obs=x)

    def train(self, n_epochs: int = 500):
        optimizer = self.optimizer()
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        guide = AutoDelta(poutine.block(self, expose=list(self.get_params(True).keys())))
        svi = SVI(self, guide, optimizer, elbo)

        with tqdm(total=n_epochs) as pbar:
            for _ in range(n_epochs):
                loss = svi.step(self.x) / self.x.numel()

                pbar.set_description(f"Loss {loss:.4e}")
                pbar.update(1)

    @staticmethod
    def conditionals(params, x):
        pyro.clear_param_store()
        zero_inflated = "z" in params
        z, counts, logits = params.get("z"), params["counts"], params["logits"]
        x = torch.tensor(x.astype(np.float32))
        n = len(counts)
        dists = (
            [dist.NegativeBinomial(c, logits=l, validate_args=False) for c, l in zip(counts, logits)]
            if "z" not in params
            else [
                dist.ZeroInflatedNegativeBinomial(c, logits=l, gate_logits=torch.tensor(_z), validate_args=False)
                for _z, c, l in zip(z, counts, logits)
            ]
        )
        if zero_inflated:
            return tuple(torch.exp(d.log_prob(x)).numpy() for d in sorted(dists, key=lambda d: -d.log_prob(0.0).item()))
        return tuple(torch.exp(d.log_prob(x)).numpy() for d in sorted(dists, key=lambda d: d.mean.item()))


def conditionals(
    X: np.ndarray,
    vi_results: Union[Dict[int, Dict[str, float]], Dict[str, float]],
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
    if any(isinstance(k, int) for k in vi_results.keys()):
        if bins is None:
            raise PreprocessingError("`vi_results` indicate binning was used, but `bins` was not provided")
        background_cond = np.ones(X.shape)
        cell_cond = np.zeros(X.shape)
        for label, params in vi_results.items():
            mask = bins == label
            conditionals = NegativeBinomialMixture.conditionals(params, X[mask])
            background_cond[mask], cell_cond[mask] = conditionals[0], conditionals[-1]
    else:
        params = vi_results
        conditionals = NegativeBinomialMixture.conditionals(params, X)
        background_cond, cell_cond = conditionals[0], conditionals[-1]

    return background_cond, cell_cond


def run_vi(
    X: np.ndarray,
    downsample: Union[int, float] = 0.01,
    n_epochs: int = 500,
    bins: Optional[np.ndarray] = None,
    zero_inflated: bool = False,
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
        nbm = NegativeBinomialMixture(_samples, zero_inflated=zero_inflated, seed=seed)
        nbm.train(n_epochs)
        results[label] = nbm.get_params()

    return results if bins is not None else results[0]
