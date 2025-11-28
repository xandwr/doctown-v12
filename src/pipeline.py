"""
End-to-end pipeline for doctown.

This ties together all components:
    1. Ingest (GitHub/ZIP → VFS)
    2. Chunk (VFS → Chunks with embeddings)
    3. Cluster (Chunks → Clusters)
    4. Apply Intent (Clusters → Structured summaries)
    5. Export (.docpack ZIP)

Usage:
    from pipeline import run_pipeline

    docpack = run_pipeline(
        source="https://github.com/user/repo",
        intent_yaml="intents/minimal.yaml",
        output_path="output/repo.docpack"
    )
"""

from pathlib import Path
from typing import Optional
from datetime import datetime

from .ingest.ingest import ingest
from .chunking import chunk_documents
from .clustering import cluster_documents
from .intent.spec import IntentSpec
from .intent.orchestrator import apply_intent
from .models import DocpackManifest, PipelineConfig, Docpack
from .docpack import save_docpack


def run_pipeline(
    source: str,
    intent_yaml: str | Path,
    output_path: Optional[str | Path] = None,
    config: Optional[PipelineConfig] = None,
    include_raw: bool = True
) -> Docpack:
    """
    Run the complete doctown pipeline.

    Args:
        source: GitHub URL, ZIP path, or directory path
        intent_yaml: Path to intent YAML spec
        output_path: Where to save .docpack (if None, returns in-memory only)
        config: Pipeline configuration (uses defaults if None)
        include_raw: Include raw source files in .docpack

    Returns:
        Complete Docpack object

    Example:
        >>> docpack = run_pipeline(
        ...     source="https://github.com/anthropics/claude-code",
        ...     intent_yaml="src/intent/examples/minimal.yaml",
        ...     output_path="output/claude-code.docpack"
        ... )
    """
    # Use default config if not provided
    if config is None:
        config = PipelineConfig()

    print(f"[1/6] Loading intent spec from {intent_yaml}...")
    intent = IntentSpec.from_yaml(intent_yaml)
    print(f"      Intent: {intent.name}")
    print(f"      {intent.description}")

    # Step 1: Ingest
    print(f"\n[2/6] Ingesting source: {source}...")
    vfs = ingest(source)
    print(f"      Loaded {len(vfs.list_files())} files ({vfs.total_bytes():,} bytes)")

    # Step 2: Chunk with embeddings
    print(f"\n[3/6] Chunking and embedding (model: {config.embedding_model})...")
    chunks = chunk_documents(
        vfs=vfs,
        chunk_size_tokens=config.chunk_size_tokens,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
        embedding_model=config.embedding_model
    )
    print(f"      Created {len(chunks)} chunks")

    total_tokens = sum(chunk.tokens for chunk in chunks)
    print(f"      Total tokens: {total_tokens:,}")

    # Step 3: Cluster
    print(f"\n[4/6] Clustering (method: {config.clustering_method}, k={config.num_clusters})...")
    clusters = cluster_documents(
        chunks=chunks,
        num_clusters=config.num_clusters,
        method=config.clustering_method
    )
    print(f"      Formed {len(clusters)} clusters")

    # Create manifest
    source_type = "github" if source.startswith("http") else "zip" if source.endswith(".zip") else "directory"
    embedding_dim = len(chunks[0].embedding) if chunks else 0

    manifest = DocpackManifest(
        created_at=datetime.utcnow(),
        source_type=source_type,
        source_identifier=source,
        intent_name=intent.name,
        intent_description=intent.description,
        file_count=len(vfs.list_files()),
        chunk_count=len(chunks),
        cluster_count=len(clusters),
        total_tokens=total_tokens,
        embedding_dim=embedding_dim,
        includes_raw_files=include_raw
    )

    # Step 4: Apply intent and generate summaries
    print(f"\n[5/6] Applying intent and generating summaries...")
    print(f"      Model: {config.summarization_model}")

    docpack = apply_intent(
        intent=intent,
        chunks=chunks,
        clusters=clusters,
        manifest=manifest,
        summarization_model=config.summarization_model
    )

    # Add raw files if requested
    if include_raw:
        raw_files = {path: vfs.get(path).data for path in vfs.list_files()}
        docpack.raw_files = raw_files

    # Step 5: Save to disk if output path provided
    if output_path:
        print(f"\n[6/6] Writing .docpack to {output_path}...")
        save_docpack(docpack, output_path)
        print(f"      Done! Docpack saved.")
    else:
        print(f"\n[6/6] Skipping save (no output path provided)")

    print(f"\n✓ Pipeline complete!")
    print(f"  - {manifest.file_count} files")
    print(f"  - {manifest.chunk_count} chunks")
    print(f"  - {manifest.cluster_count} clusters")
    print(f"  - {manifest.total_tokens:,} tokens")

    return docpack


def quick_run(
    source: str,
    intent: str = "minimal",
    output: Optional[str] = None
) -> Docpack:
    """
    Quick pipeline run with sensible defaults.

    Args:
        source: GitHub URL or ZIP path
        intent: Intent name (looks in src/intent/examples/{intent}.yaml)
        output: Output path (default: ./{repo_name}.docpack)

    Returns:
        Docpack object
    """
    # Resolve intent path
    intent_path = Path(__file__).parent / "intent" / "examples" / f"{intent}.yaml"
    if not intent_path.exists():
        raise FileNotFoundError(f"Intent '{intent}' not found at {intent_path}")

    # Auto-generate output path if not provided
    if output is None:
        if source.startswith("http"):
            # Extract repo name from GitHub URL
            repo_name = source.rstrip('/').split('/')[-1]
        else:
            # Use source filename
            repo_name = Path(source).stem
        output = f"{repo_name}.docpack"

    return run_pipeline(
        source=source,
        intent_yaml=intent_path,
        output_path=output
    )
