# ingest/zip_reader.py

import io
import zipfile
from pathlib import PurePosixPath
from .vfs import VirtualFileSystem

def sanitize_zip_path(name: str) -> str:
    """
    Prevent zip-slip, strip leading top-level folder, normalize.
    """
    parts = PurePosixPath(name).parts

    # Ignore directories
    if name.endswith('/'):
        return ""

    # Remove leading top-level dir (GitHub ZIPs are structured like repo-main/...)
    if len(parts) > 1:
        parts = parts[1:]

    # Reject weird paths
    if any(p in ("..", "/", "\\") for p in parts):
        return ""

    # Ignore empty
    if len(parts) == 0:
        return ""

    return "/".join(parts)

def load_zip_into_vfs(raw_zip: bytes) -> VirtualFileSystem:
    vfs = VirtualFileSystem()

    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        for name in z.namelist():
            clean = sanitize_zip_path(name)
            if not clean:
                continue

            data = z.read(name)
            vfs.add_file(clean, data)

    return vfs
