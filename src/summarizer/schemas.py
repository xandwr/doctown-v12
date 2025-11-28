# schemas.py
# Pydantic models for docgen output

from pydantic import BaseModel

class ChunkSummary(BaseModel):
    chunk_id: int
    summary: str

class ClusterSummary(BaseModel):
    cluster_id: int
    short: str
    long: str
    keywords: list[str]

class ProjectSummary(BaseModel):
    title: str
    overview: str
    key_components: list[str]