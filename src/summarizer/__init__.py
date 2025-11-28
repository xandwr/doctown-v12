"""
Doctown v12 Summarizer Subsystem
================================

This package provides:
- Prompt builders for semantic + structured summarization
- Pydantic schemas defining structured output formats
- Summarization functions that call Ollama with strict JSON mode

Public API:
    from doctown.summarizer import (
        summarize_cluster,
        ChunkSummary,
        ClusterSummary,
        ProjectSummary,
    )
"""

# Re-export schemas (public)
from .schemas import (
    ChunkSummary,
    ClusterSummary,
    ProjectSummary,
)

# Re-export summarization functions (public)
from .summarize import (
    summarize_cluster,
)

# Internal utilities (NOT exported)
# from .prompts import SYSTEM_SUMMARIZER, make_cluster_summary_prompt, ...

__all__ = [
    # public schemas
    "ChunkSummary",
    "ClusterSummary",
    "ProjectSummary",

    # public functions
    "summarize_cluster",
]
