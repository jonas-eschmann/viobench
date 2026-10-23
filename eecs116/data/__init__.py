"""Datasets for the challenge: the Episode container, the on-disk EuRoC/ASL layout, the camera/ISP model, and the generators."""
from .episode import Episode
from .euroc_io import read_episode, write_episode

__all__ = ["Episode", "read_episode", "write_episode"]
