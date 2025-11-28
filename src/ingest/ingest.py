# ingest/ingest.py

from .github import parse_github_url, download_github_zip
from .zip_reader import load_zip_into_vfs
from .vfs import VirtualFileSystem

def ingest(source: str) -> VirtualFileSystem:
    """
    High-level ingestion API. Supports:
      - GitHub URLs
      - local .zip files
      - local directories (optional)
    """

    if source.startswith("https://github.com/"):
        owner, repo, branch = parse_github_url(source)
        zip_bytes = download_github_zip(owner, repo, branch)
        return load_zip_into_vfs(zip_bytes)

    if source.endswith(".zip"):
        with open(source, "rb") as f:
            return load_zip_into_vfs(f.read())

    raise ValueError(f"Unsupported ingest source: {source}")
