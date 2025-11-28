"""
Intent orchestrator - applies user-defined schemas to generate structured outputs.

This module guarantees:
    project_summary = merge(clusters + metrics + intent)

The merge is deterministic and reproducible.
"""

from typing import List, Dict, Any
from ..models import Chunk, Cluster, Docpack, DocpackManifest
from ..intent.spec import IntentSpec
from ..intent.schema_builder import build_model
from ..summarizer.summarize import call_llm
from ..summarizer.prompts import (
    SYSTEM_STRUCTURED_OUTPUT,
    make_structured_cluster_prompt,
    make_structured_project_prompt
)


def summarize_cluster(
    cluster: Cluster,
    chunks: List[Chunk],
    intent: IntentSpec,
    model: str = "phi4-mini-reasoning"
) -> Dict[str, Any]:
    """
    Generate structured summary for a single cluster.

    Args:
        cluster: Cluster to summarize
        chunks: All chunks (to look up cluster members)
        intent: Intent spec with cluster_schema
        model: LLM model for summarization

    Returns:
        Structured summary matching cluster_schema
    """
    if not intent.has_cluster_schema():
        raise ValueError("Intent spec must have cluster_schema")

    # Get chunks in this cluster
    cluster_chunks = [c for c in chunks if c.chunk_id in cluster.chunk_ids]

    # Limit chunks if specified
    if len(cluster_chunks) > intent.max_chunks_per_cluster:
        cluster_chunks = cluster_chunks[:intent.max_chunks_per_cluster]

    # Build dynamic Pydantic model from schema
    ClusterModel = build_model("ClusterSummary", intent.cluster_schema)

    # Get schema for prompt
    schema_dict = intent.cluster_schema

    # Prepare chunk texts
    chunk_texts = [chunk.text for chunk in cluster_chunks]
    combined_text = "\n\n---\n\n".join(chunk_texts)

    # Build prompt
    prompt = make_structured_cluster_prompt(
        cluster_id=cluster.cluster_id,
        chunks=chunk_texts,
        schema=schema_dict
    )

    # Call LLM
    result_json = call_llm(
        system=SYSTEM_STRUCTURED_OUTPUT,
        user=prompt,
        model=model,
        schema=ClusterModel.model_json_schema()
    )

    # Validate and return
    return ClusterModel.model_validate_json(result_json).model_dump()


def summarize_project(
    clusters: List[Cluster],
    chunks: List[Chunk],
    intent: IntentSpec,
    manifest: DocpackManifest,
    model: str = "phi4-mini-reasoning"
) -> Dict[str, Any]:
    """
    Generate deterministic project-level summary.

    This guarantees: project_summary = merge(clusters + metrics + intent)

    Args:
        clusters: All clusters with summaries
        chunks: All chunks
        intent: Intent spec with project_schema
        manifest: Docpack manifest with metrics
        model: LLM model for summarization

    Returns:
        Structured summary matching project_schema
    """
    if not intent.has_project_schema():
        raise ValueError("Intent spec must have project_schema")

    # Build dynamic Pydantic model
    ProjectModel = build_model("ProjectSummary", intent.project_schema)

    # Prepare context for LLM
    # 1. Cluster summaries
    cluster_summaries_text = []
    for cluster in clusters:
        if cluster.summary:
            summary_str = f"Cluster {cluster.cluster_id}:\n"
            for key, val in cluster.summary.items():
                summary_str += f"  {key}: {val}\n"
            cluster_summaries_text.append(summary_str)

    combined_clusters = "\n\n".join(cluster_summaries_text)

    # 2. Metrics from manifest
    metrics_text = f"""
Dataset Metrics:
- Files: {manifest.file_count}
- Chunks: {manifest.chunk_count}
- Clusters: {manifest.cluster_count}
- Total Tokens: {manifest.total_tokens}
- Embedding Dimension: {manifest.embedding_dim}
"""

    # 3. Sample representative chunks (one per cluster for context)
    representative_chunks = []
    for cluster in clusters:
        if cluster.chunk_ids:
            # Take first chunk as representative
            chunk = next((c for c in chunks if c.chunk_id == cluster.chunk_ids[0]), None)
            if chunk:
                representative_chunks.append(
                    f"[Cluster {cluster.cluster_id}, File: {chunk.file_path}]\n{chunk.text[:500]}..."
                )

    samples_text = "\n\n---\n\n".join(representative_chunks)

    # Build prompt
    prompt = make_structured_project_prompt(
        cluster_summaries=combined_clusters,
        metrics=metrics_text,
        representative_samples=samples_text,
        schema=intent.project_schema
    )

    # Call LLM
    result_json = call_llm(
        system=SYSTEM_STRUCTURED_OUTPUT,
        user=prompt,
        model=model,
        schema=ProjectModel.model_json_schema()
    )

    # Validate and return
    return ProjectModel.model_validate_json(result_json).model_dump()


def apply_intent(
    intent: IntentSpec,
    chunks: List[Chunk],
    clusters: List[Cluster],
    manifest: DocpackManifest,
    summarization_model: str = "phi4-mini-reasoning"
) -> Docpack:
    """
    Apply intent spec to generate all structured outputs and build complete Docpack.

    This is the main entry point for the orchestrator.

    Args:
        intent: Intent specification
        chunks: All chunks with embeddings and cluster assignments
        clusters: All clusters
        manifest: Docpack manifest with metrics
        summarization_model: LLM model for summaries

    Returns:
        Complete Docpack with all summaries
    """
    # 1. Summarize clusters if schema provided
    if intent.has_cluster_schema():
        for cluster in clusters:
            cluster.summary = summarize_cluster(
                cluster=cluster,
                chunks=chunks,
                intent=intent,
                model=summarization_model
            )

    # 2. Summarize project if schema provided and allowed
    project_summary = {}
    if intent.has_project_schema() and intent.allow_global_summary:
        project_summary = summarize_project(
            clusters=clusters,
            chunks=chunks,
            intent=intent,
            manifest=manifest,
            model=summarization_model
        )

    # 3. Build Docpack
    docpack = Docpack(
        manifest=manifest,
        intent_spec=intent.to_dict(),
        chunks=chunks,
        clusters=clusters,
        project_summary=project_summary,
        raw_files=None  # Set externally if needed
    )

    return docpack
