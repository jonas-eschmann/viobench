"""Episodes and where they come from.

Two providers yield `Episode` objects:
- `storage.episodes(roots)`            episode archives (or unzipped directories) on disk, frames streamed
- `generate.simulation.simulate(...)`  fresh episodes from HyperDrone, frames spooled to a temporary directory (NVIDIA GPU)
The runner and the training example consume either. `sources.episodes(sources)` is how the CLI, the runner and the
server name episodes: paths and conta references (hashes of archives, `viobench.episodes`) in any mix, the references
downloaded into the conta cache and read from there by the first provider.
"""
from .episode import Episode
from .storage import episodes, list_episodes, read_episode, write_episode

__all__ = ["Episode", "episodes", "list_episodes", "read_episode", "write_episode"]
