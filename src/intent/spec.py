from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


class IntentSpec(BaseModel):
    """
    Declarative specification for what structured outputs to generate.

    Users define schemas for cluster and project summaries, and the
    system dynamically generates Pydantic models to validate LLM output.
    """

    name: str = Field(..., description="Unique identifier for this intent preset")
    description: str = Field(..., description="Human-readable purpose")

    # User-defined structured outputs
    project_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Schema for project-level summary (top-level output)"
    )
    cluster_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Schema for cluster-level summaries"
    )
    chunk_schema: Optional[Dict[str, Any]] = Field(
        None,
        description="Schema for individual chunk summaries (rarely used)"
    )

    # Optional behavioral controls
    max_chunks_per_cluster: int = Field(
        10,
        description="Maximum chunks to include per cluster summary"
    )
    allow_cross_file_inference: bool = Field(
        False,
        description="Allow model to infer relationships across files"
    )
    allow_global_summary: bool = Field(
        True,
        description="Generate project-level summary"
    )

    @field_validator('project_schema', 'cluster_schema', 'chunk_schema')
    @classmethod
    def validate_schema(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ensure schema is valid if provided."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("Schema must be a dictionary")
        if v is not None and len(v) == 0:
            raise ValueError("Schema cannot be empty")
        return v

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "IntentSpec":
        """
        Load an IntentSpec from a YAML file.

        Args:
            yaml_path: Path to the YAML file

        Returns:
            IntentSpec: Parsed and validated intent spec

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid or fails validation
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Intent spec not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML: expected dict, got {type(data)}")

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode='json')

    def has_cluster_schema(self) -> bool:
        """Check if cluster schema is defined."""
        return self.cluster_schema is not None

    def has_project_schema(self) -> bool:
        """Check if project schema is defined."""
        return self.project_schema is not None