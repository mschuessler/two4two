"""utility functions."""

from typing import Any, Dict, Sequence, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


T = TypeVar('T')


class discrete():
    """Wrapper around ``scypi.stats.rv_discrete`` to support more than ints.

    Attrs:
        values: The values of the discrete distribution.
        rv_discrete: The ``scypi.stats.rv_discrete`` distributon.
    """
    def __init__(
            self,
            value_to_probs: Dict[T, float],
            **stats_kwargs: Dict[str, Any]):
        """A discrete distribution with any values.

        Args:
            value_to_probs: A mapping of the distribution's values to probabilities.
            **stats_kwargs: Passed on to ``scipy.stats.rv_discrete``.
        """
        self.values = list(value_to_probs.keys())
        value_indicies = list(range(len(self.values)))
        probs = list(value_to_probs.values())
        self.rv_discrete = scipy.stats.rv_discrete(
            values=[value_indicies, probs], **stats_kwargs)

    def pmf(self,
            k: Union[T, Sequence[T]],
            *args: Sequence[Any],
            **kwargs: Dict[str, Any]
            ) -> Sequence[float]:
        """Probability mass function."""
        return np.exp(self.logpmf(k, *args, **kwargs))

    def logpmf(self,
               k: Union[T, Sequence[T]],
               *args: Sequence[Any],
               **kwargs: Dict[str, Any]
               ) -> Sequence[float]:
        """Log Probability mass function."""
        if k in self.values:  # a single k
            return self.rv_discrete.logpmf(self.values.index(k))
        else:  # a sequence of k's
            return self.rv_discrete.logpmf([self.values.index(k_item) for k_item in k])

    def rvs(self,
            *args: Sequence[Any],
            **kwargs: Dict[str, Any]
            ) -> T:
        """Samples from distribution."""
        return self.values[self.rv_discrete.rvs(*args, **kwargs)]


def truncated_normal(mean: float = 0,
                     std: float = 1,
                     lower: float = -3,
                     upper: float = 3
                     ) -> scipy.stats.truncnorm:
    """Wrapper around ``scipy.stats.truncnorm``.

    Args:
        mean: the mean of the normal distribution.
        std: the standard derivation of the normal distribution.
        lower: lower truncation.
        upper: upper truncation.
    """
    return scipy.stats.truncnorm((lower - mean) / std, (upper - mean) / std,
                                 loc=mean, scale=std)


class ColorGenerator():

    def __init__(self, palette=None):
        self.cmap = plt.get_cmap(palette)

    def get_color(self, val=None):
        if val is not None:
            return self.cmap(val)
        elif type(self.cmap) == list:
            cm = np.random.choice(self.cmap)
            return cm(np.random.uniform())
        else:
            return self.cmap(np.random.uniform())

    def get_random_color(self):
        color = np.random.uniform(size=3)
        return tuple(color) + (1,)


def splitStickyStretchy(params, num_samples_per_class=None):
    return [p for p in params if p.obj_name == 'sticky'][:num_samples_per_class], \
        [p for p in params if p.obj_name == 'stretchy'][:num_samples_per_class]
