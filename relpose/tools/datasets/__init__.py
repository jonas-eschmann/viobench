"""Datasets for the challenge: the Episode container, the on-disk EuRoC/ASL layout, the camera/ISP model, and the generators."""
from .episode import Episode
from .storage import list_episodes, read_episode, write_episode

__all__ = ["Episode", "list_episodes", "read_episode", "write_episode"]
