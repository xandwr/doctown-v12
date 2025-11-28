"""
Core data models for the doctown pipeline.

These models define the structure of chunks, clusters, and the final .docpack output.
All models are Pydantic for validation and serialization.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Chunk(BaseModel):
    """A single text chunk from the ingested documents."""

    chunk_id: int = Field(..., description="Unique identifier for this chunk")
    file_path: str = Field(..., description="Source file path (relative, from VFS)")
    start_line: int = Field(..., description="Starting line number in source file")
    end_line: int = Field(..., description="Ending line number in source file")
    tokens: int = Field(..., description="Token count for this chunk")
    text: str = Field(..., description="The actual text content")
    embedding: List[float] = Field(..., description="Embedding vector for this chunk")
    cluster_id: Optional[int] = Field(None, description="Assigned cluster ID (set during clustering)")

    class Config:
        frozen = False  # Allow mutation for cluster_id assignment


class Cluster(BaseModel):
    """A semantic cluster of related chunks."""

    cluster_id: int = Field(..., description="Unique identifier for this cluster")
    chunk_ids: List[int] = Field(..., description="IDs of chunks in this cluster")
    centroid: Optional[List[float]] = Field(None, description="Cluster centroid embedding")
    summary: Optional[Dict[str, Any]] = Field(None, description="Structured summary (schema from intent)")

    class Config:
        frozen = False  # Allow mutation for summary assignment


class DocpackManifest(BaseModel):
    """Metadata about a .docpack archive."""

    # Generation metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this docpack was created")
    doctown_version: str = Field("0.1.0", description="Version of doctown that created this")

    # Source metadata
    source_type: str = Field(..., description="Type of source: 'github', 'zip', 'directory'")
    source_identifier: str = Field(..., description="GitHub URL, ZIP path, or directory path")

    # Processing metadata
    intent_name: str = Field(..., description="Name of the intent spec used")
    intent_description: str = Field(..., description="Description of the intent")

    # Metrics (these guarantee deterministic merge)
    file_count: int = Field(..., description="Number of files ingested")
    chunk_count: int = Field(..., description="Number of chunks created")
    cluster_count: int = Field(..., description="Number of clusters formed")
    total_tokens: int = Field(..., description="Total tokens across all chunks")
    embedding_dim: int = Field(..., description="Dimensionality of embeddings")

    # Flags
    includes_raw_files: bool = Field(True, description="Whether raw/ directory is included")


class Docpack(BaseModel):
    """
    Complete in-memory representation of a .docpack.

    This is the canonical structure that gets serialized to ZIP.
    The project_summary is deterministically derived from:
        - All cluster summaries
        - Manifest metrics
        - Intent spec requirements
    """

    manifest: DocpackManifest
    intent_spec: Dict[str, Any] = Field(..., description="The full intent spec as dict")
    chunks: List[Chunk] = Field(..., description="All text chunks with embeddings")
    clusters: List[Cluster] = Field(..., description="All clusters with summaries")
    project_summary: Dict[str, Any] = Field(..., description="Top-level structured summary")
    raw_files: Optional[Dict[str, bytes]] = Field(None, description="Original files {path: bytes}")

    def get_chunk(self, chunk_id: int) -> Optional[Chunk]:
        """Retrieve a chunk by ID."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_cluster(self, cluster_id: int) -> Optional[Cluster]:
        """Retrieve a cluster by ID."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def get_chunks_for_cluster(self, cluster_id: int) -> List[Chunk]:
        """Get all chunks belonging to a cluster."""
        cluster = self.get_cluster(cluster_id)
        if not cluster:
            return []
        return [self.get_chunk(cid) for cid in cluster.chunk_ids if self.get_chunk(cid)]


class PipelineConfig(BaseModel):
    """Configuration for the full pipeline execution."""

    # Chunking parameters
    chunk_size_tokens: int = Field(500, description="Target tokens per chunk")
    chunk_overlap_tokens: int = Field(50, description="Overlap between consecutive chunks")

    # Clustering parameters
    num_clusters: int = Field(10, description="Target number of clusters")
    clustering_method: str = Field("kmeans", description="Clustering algorithm: kmeans, hierarchical, etc")

    # Embedding parameters
    embedding_model: str = Field("nomic-embed-text", description="Ollama embedding model")

    # Summarization parameters
    summarization_model: str = Field("phi4-mini-reasoning", description="Ollama model for summaries")

    # Output parameters
    include_raw_files: bool = Field(True, description="Include raw/ directory in .docpack")
