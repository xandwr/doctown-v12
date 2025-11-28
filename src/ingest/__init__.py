# ingest/__init__.py

from .ingest import ingest
from .vfs import VirtualFileSystem, VFSFile

__all__ = ["ingest", "VirtualFileSystem", "VFSFile"]
