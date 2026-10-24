"""Datasets: episodes and where they come from.

A *Dataset* is simply an iterable of `Episode`s. Two providers implement it:
- `storage.episodes(roots)`            episode directories on disk (HyperDrone-generated or EuRoC-converted), frames streamed
- `generate.simulation.simulate(...)`  fresh episodes from HyperDrone, never written to disk (GPU; `pip install relpose[sim]`)
The runner and the training example consume either; the server uses the first.
"""
from .episode import Episode
from .storage import episodes, list_episodes, read_episode, write_episode

__all__ = ["Episode", "episodes", "list_episodes", "read_episode", "write_episode"]
