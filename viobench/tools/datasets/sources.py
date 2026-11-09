"""Episode sources: how the CLI, the runner and the server name episodes — paths, conta references, or a mix.

A source is
- a path: an episode archive, an unzipped episode, or a directory of them (the file-system provider, `storage.episodes`);
- a conta reference to an episode archive: its sha1 (40 hexadecimal digits), "conta:<sha1>", or an entry
  {"description", "hash"} of a set in `viobench.episodes`. conta downloads the archive once into its cache
  (CONTA_CACHE, default ~/.cache/conta; with CONTA_ROOT=<store> it reads <store>/data/<sha1> instead, without network),
  and the archive is read in place from there like one on disk.

An episode is its hash, the sha1 of its archive (`Episode.hash`), however it is named: a reference, or a path to the
archive (hashed when read); an unzipped episode has none. Presets in `viobench.episodes.SETS` are named collections of
hashes, and their descriptions label conta's logs. An existing path wins over a bare hash of the same spelling.
"""
from pathlib import Path

from ... import episodes as published
from .storage import episodes as stored_episodes
from .storage import read_episode

PREFIX = "conta:"
HEX_DIGITS = frozenset("0123456789abcdef")
HELP = ("episodes (.zip or unzipped), directories of them, or conta hashes of episode archives (sha1 or conta:<sha1>); "
        "default: the public set (viobench.episodes.public)")


def reference_hash(source):
    """The sha1 that a conta reference names (normalised to lower case), or None if source is not one."""
    if isinstance(source, dict):
        source = source.get("hash")
    if not isinstance(source, str):
        return None
    digest = source.removeprefix(PREFIX).lower()
    return digest if len(digest) == 40 and set(digest) <= HEX_DIGITS else None


def argument(reference):
    """A conta reference as a command-line source: "conta:<sha1>"."""
    digest = reference_hash(reference)
    if digest is None:
        raise ValueError(f"not a conta reference: {reference!r}")
    return PREFIX + digest


def episodes(sources, frames="lazy"):
    """The episodes of sources, in order, read as the iterator advances (frames: as `storage.read_episode`).

    The sources are checked when called: ValueError for a malformed conta reference, FileNotFoundError for a source that
    is neither an existing path nor a hash. A referenced archive is resolved when the iterator reaches it, downloaded if
    the cache lacks it (conta.ContaError if it cannot be), so a run that stops early downloads only what it read."""
    return _read([(source, _classify(source)) for source in _as_list(sources)], frames)


def resolve(sources):
    """The local path of each source, in order: a path as given, a conta reference as its archive in conta's cache. The
    references are resolved now, together (one walk over the conta index), downloading what the cache lacks; raises as
    `episodes` does, and conta.ContaError for a reference that cannot be resolved."""
    sources = _as_list(sources)
    digests = [_classify(source) for source in sources]
    wanted = list(dict.fromkeys(digest for digest in digests if digest is not None))
    paths = dict(zip(wanted, _resolve_all(wanted, _references()), strict=True))
    return [Path(source) if digest is None else paths[digest] for source, digest in zip(sources, digests)]


def add_argument(parser):
    """The `episodes` argument of `viobench evaluate` and `viobench debug`; default to the public preset."""
    parser.add_argument("episodes", nargs="*", help=HELP)


def from_arguments(sources, frames="lazy"):
    """The episodes of the positional arguments (add_argument)."""
    return episodes(sources or published.public, frames)


def _as_list(sources):
    return [sources] if isinstance(sources, (str, Path, dict)) else list(sources)


def _classify(source):
    """The sha1 of a conta reference, None for a path."""
    if isinstance(source, dict) or (isinstance(source, str) and source.startswith(PREFIX)):
        digest = reference_hash(source)
        if digest is None:
            raise ValueError(f"malformed conta reference {source!r}: expected a sha1 (40 hexadecimal digits)")
        return digest
    if Path(source).exists():
        return None
    digest = reference_hash(source)
    if digest is None:
        raise FileNotFoundError(f"{source}: neither an episode, a directory of episodes, nor a conta hash (sha1 or conta:<sha1>)")
    return digest


def _references():
    """Catalog references supply log labels when an episode is named only by its hash."""
    return {entry["hash"]: entry for entries in published.SETS.values() for entry in entries}


def _resolve_all(digests, listed):
    """The cached archives of digests; a listed one is resolved by its set's entry, so conta logs its description."""
    if not digests:
        return []
    import conta  # not at the top: urllib and ssl stay out of the CLI's startup

    return [Path(path) for path in conta.resolve_all([listed.get(digest, digest) for digest in digests])]


def _read(classified, frames):
    listed = _references()
    for source, digest in classified:
        if digest is None:
            yield from stored_episodes([source], frames=frames)
        else:
            [path] = _resolve_all([digest], listed)
            yield read_episode(path, frames=frames, digest=digest)
