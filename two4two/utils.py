"""utility functions."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import scipy.stats


RGBAColor = Tuple[float, float, float, float]

QUARTER_CIRCLE = (-np.pi / 4, np.pi / 4)
# HALF_CIRCLE = (-np.pi / 4, np.pi / 4)
FULL_CIRCLE = (-np.pi, np.pi)

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
        if supports_iteration(k):
            return self.rv_discrete.logpmf(
                [self.values.index(k_item) for k_item in k])      # type: ignore
        # a single k
        elif k in self.values:  # type: ignore
            return self.rv_discrete.logpmf(self.values.index(k))  # type: ignore
        else:
            raise ValueError(f'Neither iterable nor in self.values: {k}')

    def rvs(self,
            *args: Sequence[Any],
            **kwargs: Dict[str, Any]
            ) -> T:
        """Samples from distribution."""
        return self.values[self.rv_discrete.rvs(*args, **kwargs)]


def numpy_to_python_scalar(x: np.ndarray) -> Union[int, float]:
    """Returns ``x`` as python scalar."""
    if isinstance(x, np.floating):
        return float(x)
    elif issubclass(x.dtype.type, np.integer):
        return int(x)
    else:
        raise ValueError(f"Cannot convert {x} to int or float.")


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


def supports_iteration(value: Union[Any, Sequence[Any]]) -> bool:
    """Returns ``True`` if the ``value`` supports iterations."""
    try:
        for _ in value:
            return True
    except TypeError:
        pass
    return False


def split_class(module_dot_class: str) -> Tuple[str, str]:
    """Splits ``"my.module.MyClass"`` into ``("my.module", "MyClass")``."""
    parts = module_dot_class.split('.')
    module = '.'.join(parts[:-1])
    cls_name = parts[-1]
    return module, cls_name


def import_class(module: str, cls_name: str) -> Type:
    """Returns the class given as string."""
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    return cls


def get(maybe_none: Optional[T], default: T) -> T:
    """Returns either the given value if not ``None`` or a default value.

    Useful helper function to deconstruct Optional values in a typesafe way.
    """
    if maybe_none is not None:
        return maybe_none
    else:
        return default
