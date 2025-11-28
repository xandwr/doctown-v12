# ingest/github.py

import requests

def parse_github_url(url: str):
    """
    Accepts:
      https://github.com/user/repo
      https://github.com/user/repo/
      https://github.com/user/repo/tree/branch
      https://github.com/user/repo/blob/branch/path
    Normalizes to: (owner, repo, branch)
    """
    parts = url.rstrip("/").split("/")
    owner = parts[3]
    repo  = parts[4]

    # Try to detect custom branch
    if len(parts) > 6 and parts[5] == "tree":
        branch = parts[6]
    else:
        branch = "main"

    return owner, repo, branch

def download_github_zip(owner: str, repo: str, branch: str) -> bytes:
    url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    return resp.content
