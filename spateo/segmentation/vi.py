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
from torch.distributions.utils import logits_to_probs, probs_to_logits
from tqdm import tqdm

from ..errors import SegmentationError


class NegativeBinomialMixture(PyroModule):
    def __init__(
        self,
        x: np.ndarray,
        n: int = 2,
        n_init: int = 5,
        w: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        var: Optional[np.ndarray] = None,
        zero_inflated: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if not ((w is None) == (mu is None) and (w is None) == (var is None)):
            raise SegmentationError("All or none of `w`, `mu`, `var` must be provided.")
        if (w is not None) and (n != len(w) or n != len(mu) or n != len(var)):
            raise SegmentationError(f"`w`, `mu`, `var` must have length {n}.")

        if seed is not None:
            torch.manual_seed(seed)
        self.zero_inflated = zero_inflated
        self.x = torch.tensor(x.astype(np.float32))
        self.n = n
        self.scale = torch.median(self.x[self.x > 0])

        if w is not None:
            self.init_mean_variance(w, mu, var)
        else:
            self.init_best_params(n_init)
        self.__optimizer = None

    def assignment(self, train=False):
        params = self.get_params(train)
        w = params["w"]
        return dist.Categorical(logits=w)

    def dist(self, assignment, train=False):
        params = self.get_params(train)
        counts, logits = params["counts"], params["logits"]
        z = params.get("z", probs_to_logits(torch.zeros(self.n)))
        return dist.ZeroInflatedNegativeBinomial(
            counts[assignment], logits=logits[assignment], gate_logits=z[assignment], validate_args=False
        )

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

    def init_mean_variance(self, w, mu, var):
        self.w = PyroParam(probs_to_logits(torch.tensor(w).float()))

        counts = torch.zeros(self.n)
        logits = torch.zeros(self.n)
        for i, (m, v) in enumerate(zip(mu, var)):
            prob = 1 - m / v
            logits[i] = probs_to_logits(torch.tensor(prob), is_binary=True).item()

            # Inverse softplus for counts
            counts[i] = (m * (1 - prob) / prob) / self.scale
            if counts[i] <= 20:
                counts[i] = torch.log(torch.exp(counts[i]) - 1)

        self.counts = PyroParam(counts)
        self.logits = PyroParam(logits)

        # Is there a better way to initialize the dropout param?
        if self.zero_inflated:
            self.z = PyroParam(probs_to_logits(torch.zeros(self.n).float(), is_binary=True))

    def optimizer(self):
        if self.__optimizer is None:
            self.__optimizer = Adam({"lr": 0.01})
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
    def conditionals(params, x, use_weights=False):
        pyro.clear_param_store()
        zero_inflated = "z" in params
        w, counts, logits = params["w"], params["counts"], params["logits"]
        n = len(w)
        z = params.get("z", probs_to_logits(torch.zeros(n)))
        x = torch.tensor(x.astype(np.float32))
        dists = [
            dist.ZeroInflatedNegativeBinomial(c, logits=l, gate_logits=torch.tensor(_z), validate_args=False)
            for _z, c, l in zip(z, counts, logits)
        ]
        # As of 2022/05/14, Pyro's ZeroInflatedNegativeBinomial model has a bug when calculating the mean of the
        # distribution when it was initialized with gate_logits.
        means = [(1 - logits_to_probs(dist.gate_logits, is_binary=True)) * dist.base_dist.mean for dist in dists]

        weights = dist.Categorical(logits=torch.tensor(w)).probs.numpy()
        conds = []
        for i in sorted(range(len(dists)), key=lambda i: means[i]):
            cond = torch.exp(dists[i].log_prob(x)).numpy()
            if use_weights:
                cond *= weights[i]
            conds.append(cond)
        return tuple(conds)


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
        SegmentationError: If `em_results` is a dictionary but `bins` was not
            provided.
    """
    if "counts" not in vi_results:
        if bins is None:
            raise SegmentationError("`vi_results` indicate binning was used, but `bins` was not provided")
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
    params: Union[Dict[str, Tuple[float, float]], Dict[int, Dict[str, Tuple[float, float]]]] = dict(
        w=(0.5, 0.5), mu=(10.0, 300.0), var=(20.0, 400.0)
    ),
    zero_inflated: bool = False,
    seed: Optional[int] = None,
) -> Union[
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
]:
    """Run negative binomial mixture variational inference.

    Args:
        X:
        downsample:
        n_epochs:
        bins:
        params:
        zero_inflated:
        seed:

    Returns:
    """
    samples = {}  # key 0 when bins = None
    if bins is not None:
        for label in np.unique(bins):
            if label > 0:
                samples[label] = X[bins == label]
                _params = params.get(label, params)
                if set(_params.keys()) != {"w", "mu", "var"}:
                    raise SegmentationError("`params` must contain exactly the keys `w`, `mu`, `var`.")
    else:
        samples[0] = X.flatten()
        if set(params.keys()) != {"w", "mu", "var"}:
            raise SegmentationError("`params` must contain exactly the keys `w`, `mu`, `var`.")

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
        nbm = NegativeBinomialMixture(_samples, zero_inflated=zero_inflated, seed=seed, **params.get(label, params))
        nbm.train(n_epochs)
        results[label] = nbm.get_params()

    return results if bins is not None else results[0]
