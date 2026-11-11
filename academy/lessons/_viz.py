"""Matplotlib figure builders shared by the Level 1 lessons. (Leading underscore: not a lesson.)"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .. import theme as T


def _xrange(dists, lo=0.001, hi=0.999):
    a = min(d.ppf(lo) for d in dists)
    b = max(d.ppf(hi) for d in dists)
    pad = (b - a) * 0.05
    return max(0.0, a - pad), min(1.0, b + pad)


def beta_update_fig(a0: float, b0: float, k: int, n: int):
    prior, like, post = stats.beta(a0, b0), stats.beta(k + 1, n - k + 1), stats.beta(a0 + k, b0 + n - k)
    lo, hi = _xrange([prior, like, post])
    x = np.linspace(lo, hi, 500)
    fig, ax = plt.subplots()
    ax.plot(x, prior.pdf(x), color=T.MUTED, lw=2, ls="--", label=f"Prior Beta({a0:g}, {b0:g})")
    ax.plot(x, like.pdf(x), color=T.AMBER, lw=2, label=f"Data only ({k} of {n})")
    ax.plot(x, post.pdf(x), color=T.BLUE, lw=2.6, label="Posterior")
    ax.fill_between(x, post.pdf(x), color=T.BLUE, alpha=0.18)
    ax.axvline(post.mean(), color=T.BLUE, lw=1, ls=":")
    ax.set_xlabel("True rate θ"), ax.set_ylabel("Plausibility"), ax.set_yticks([])
    ax.legend(loc="upper left")
    return fig


def multi_posterior_fig(priors: dict, k: int, n: int):
    dists = {name: stats.beta(a + k, b + n - k) for name, (a, b) in priors.items()}
    lo, hi = _xrange(list(dists.values()))
    x = np.linspace(lo, hi, 500)
    fig, ax = plt.subplots()
    for (name, d), c in zip(dists.items(), [T.AMBER, T.BLUE, T.ROSE]):
        ax.plot(x, d.pdf(x), color=c, lw=2.2, label=f"{name}: mean {d.mean():.3f}")
    ax.axvline(k / n, color=T.MUTED, lw=1.2, ls="--", label=f"Raw rate {k / n:.3f}")
    ax.set_xlabel("True rate θ"), ax.set_ylabel("Plausibility"), ax.set_yticks([])
    ax.legend(loc="upper right")
    return fig


def likelihood_fig(k: int, n: int, thetas=(0.6, 0.8, 0.9)):
    x = np.linspace(0.001, 0.999, 600)
    lik = stats.binom.pmf(k, n, x)
    mle = k / n
    fig, ax = plt.subplots()
    ax.plot(x, lik, color=T.BLUE, lw=2.4)
    ax.fill_between(x, lik, color=T.BLUE, alpha=0.15)
    for th, c in zip(thetas, [T.AMBER, T.GREEN, T.ROSE]):
        v = stats.binom.pmf(k, n, th)
        ax.plot([th], [v], "o", color=c, ms=8, label=f"θ = {th}: L = {v:.3f}")
    ax.axvline(mle, color=T.MUTED, ls="--", lw=1.2, label=f"Best fit (MLE) = {mle:.2f}")
    ax.set_xlabel("Candidate true rate θ"), ax.set_ylabel(f"P(data | θ), data = {k} of {n}")
    ax.legend(loc="upper left")
    return fig


def sequential_fig(a0: float, b0: float, weeks: list, upto: int):
    x = np.linspace(0.0, 0.4, 500)
    fig, ax = plt.subplots()
    ax.plot(x, stats.beta(a0, b0).pdf(x), color=T.MUTED, ls="--", lw=2, label="Prior (before season)")
    a, b = a0, b0
    shades = [T.CYAN, T.AMBER, T.GREEN, T.BLUE]
    for w in range(upto):
        k, n = weeks[w]
        a, b = a + k, b + n - k
        ax.plot(x, stats.beta(a, b).pdf(x), color=shades[w % 4], lw=2.2 if w < upto - 1 else 3,
                label=f"After week {w + 1}")
    ax.set_xlabel("True pressure rate θ"), ax.set_ylabel("Plausibility"), ax.set_yticks([])
    ax.legend(loc="upper right")
    return fig
