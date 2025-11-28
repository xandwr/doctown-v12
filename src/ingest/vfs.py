# ingest/vfs.py

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class VFSFile:
    path: str      # normalized virtual path: "src/lib.rs"
    data: bytes    # file contents
    size: int      # convenience field

class VirtualFileSystem:
    """
    A safe, flattened in-memory filesystem for ingested archives.
    No directory traversal, no absolute paths, no hidden dotfiles unless allowed.
    """

    def __init__(self):
        self.files: Dict[str, VFSFile] = {}

    def add_file(self, path: str, data: bytes):
        self.files[path] = VFSFile(path=path, data=data, size=len(data))

    def get(self, path: str) -> VFSFile | None:
        return self.files.get(path)

    def list_files(self) -> List[str]:
        return list(self.files.keys())

    def total_bytes(self) -> int:
        return sum(f.size for f in self.files.values())

    def __len__(self):
        return len(self.files)
