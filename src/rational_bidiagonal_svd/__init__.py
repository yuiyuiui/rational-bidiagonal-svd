"""Rational filters for selected bidiagonal singular directions."""
from .api import ResourceLimitError, SVDResult, UnitDirection, selected_svd
from ._compiler import BackendUnavailable

__version__ = '0.1.0'
__all__ = ['selected_svd', 'SVDResult', 'UnitDirection', 'BackendUnavailable', 'ResourceLimitError']
