"""
.docpack file format implementation.

A .docpack is a ZIP archive with the following structure:

    manifest.json          - DocpackManifest metadata
    intent.yaml            - The intent spec used
    chunks/                - Individual chunk JSON files
        chunk_0000.json
        chunk_0001.json
        ...
    clusters/              - Individual cluster JSON files
        cluster_00.json
        cluster_01.json
        ...
    project_summary.json   - Top-level structured summary
    raw/                   - (Optional) Original source files
        src/main.py
        README.md
        ...

All JSON uses Pydantic serialization for type safety.
"""

import json
import zipfile
from pathlib import Path
from typing import Optional, BinaryIO
import yaml

from .models import Docpack, DocpackManifest, Chunk, Cluster


class DocpackWriter:
    """Writes a Docpack to a .docpack ZIP file."""

    def __init__(self, docpack: Docpack):
        self.docpack = docpack

    def write(self, output_path: str | Path) -> None:
        """
        Write the docpack to a ZIP file.

        Args:
            output_path: Path to the output .docpack file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Write manifest
            self._write_json(zf, "manifest.json", self.docpack.manifest.model_dump(mode='json'))

            # Write intent spec
            self._write_yaml(zf, "intent.yaml", self.docpack.intent_spec)

            # Write chunks (one file per chunk for easy inspection)
            for chunk in self.docpack.chunks:
                chunk_filename = f"chunks/chunk_{chunk.chunk_id:04d}.json"
                self._write_json(zf, chunk_filename, chunk.model_dump(mode='json'))

            # Write clusters
            for cluster in self.docpack.clusters:
                cluster_filename = f"clusters/cluster_{cluster.cluster_id:02d}.json"
                self._write_json(zf, cluster_filename, cluster.model_dump(mode='json'))

            # Write project summary
            self._write_json(zf, "project_summary.json", self.docpack.project_summary)

            # Write raw files if included
            if self.docpack.manifest.includes_raw_files and self.docpack.raw_files:
                for file_path, file_bytes in self.docpack.raw_files.items():
                    raw_path = f"raw/{file_path}"
                    zf.writestr(raw_path, file_bytes)

    @staticmethod
    def _write_json(zf: zipfile.ZipFile, path: str, data: dict) -> None:
        """Write JSON data to the zip file."""
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        zf.writestr(path, json_str)

    @staticmethod
    def _write_yaml(zf: zipfile.ZipFile, path: str, data: dict) -> None:
        """Write YAML data to the zip file."""
        yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        zf.writestr(path, yaml_str)


class DocpackReader:
    """Reads a .docpack ZIP file back into a Docpack object."""

    def __init__(self, docpack_path: str | Path):
        self.docpack_path = Path(docpack_path)
        if not self.docpack_path.exists():
            raise FileNotFoundError(f"Docpack file not found: {self.docpack_path}")

    def read(self) -> Docpack:
        """
        Read and reconstruct the Docpack from the ZIP file.

        Returns:
            Docpack: The reconstructed docpack object
        """
        with zipfile.ZipFile(self.docpack_path, 'r') as zf:
            # Read manifest
            manifest_data = self._read_json(zf, "manifest.json")
            manifest = DocpackManifest(**manifest_data)

            # Read intent spec
            intent_spec = self._read_yaml(zf, "intent.yaml")

            # Read chunks
            chunks = []
            chunk_files = [name for name in zf.namelist() if name.startswith("chunks/")]
            for chunk_file in sorted(chunk_files):
                chunk_data = self._read_json(zf, chunk_file)
                chunks.append(Chunk(**chunk_data))

            # Read clusters
            clusters = []
            cluster_files = [name for name in zf.namelist() if name.startswith("clusters/")]
            for cluster_file in sorted(cluster_files):
                cluster_data = self._read_json(zf, cluster_file)
                clusters.append(Cluster(**cluster_data))

            # Read project summary
            project_summary = self._read_json(zf, "project_summary.json")

            # Read raw files if present
            raw_files = None
            if manifest.includes_raw_files:
                raw_files = {}
                raw_file_names = [name for name in zf.namelist() if name.startswith("raw/")]
                for raw_file in raw_file_names:
                    # Strip "raw/" prefix to get original path
                    original_path = raw_file[4:]  # len("raw/") == 4
                    raw_files[original_path] = zf.read(raw_file)

            return Docpack(
                manifest=manifest,
                intent_spec=intent_spec,
                chunks=chunks,
                clusters=clusters,
                project_summary=project_summary,
                raw_files=raw_files
            )

    @staticmethod
    def _read_json(zf: zipfile.ZipFile, path: str) -> dict:
        """Read and parse JSON from the zip file."""
        data = zf.read(path)
        return json.loads(data)

    @staticmethod
    def _read_yaml(zf: zipfile.ZipFile, path: str) -> dict:
        """Read and parse YAML from the zip file."""
        data = zf.read(path)
        return yaml.safe_load(data)


def save_docpack(docpack: Docpack, output_path: str | Path) -> None:
    """
    Convenience function to save a docpack to a file.

    Args:
        docpack: The Docpack object to save
        output_path: Path to the output .docpack file
    """
    writer = DocpackWriter(docpack)
    writer.write(output_path)


def load_docpack(docpack_path: str | Path) -> Docpack:
    """
    Convenience function to load a docpack from a file.

    Args:
        docpack_path: Path to the .docpack file

    Returns:
        Docpack: The loaded docpack object
    """
    reader = DocpackReader(docpack_path)
    return reader.read()
