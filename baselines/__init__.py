"""
baselines package for OceanEmbed.
"""

from baselines.climatology import ClimatologyBaseline
from baselines.ridge import ChunkedRidgeBaseline

__all__ = [
    "ClimatologyBaseline",
    "ChunkedRidgeBaseline",
]
