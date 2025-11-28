"""
Dynamic Pydantic model builder from user-defined schemas.

Converts YAML schema definitions into runtime Pydantic models for
structured LLM output validation.
"""

from pydantic import BaseModel, create_model, Field
from typing import Any, Dict, Optional, Type, get_origin, get_args


def infer_type(value: Any) -> Type:
    """
    Infer Python type from schema value.

    Supports:
    - Primitives: "str", "int", "float", "bool"
    - Lists: ["str"], ["int"], etc.
    - Dicts: {"*": ["str"]} for flexible key-value pairs
    - Nested schemas: {nested dict}

    Args:
        value: Schema value from YAML

    Returns:
        Python type annotation
    """
    # Handle list types: ["str"] -> List[str]
    if isinstance(value, list):
        if len(value) == 0:
            return list  # Untyped list
        element_type = infer_type(value[0])
        return list[element_type]

    # Handle dict types: {"*": ["str"]} -> Dict[str, List[str]]
    if isinstance(value, dict):
        if "*" in value:
            # Flexible key-value dict
            value_type = infer_type(value["*"])
            return dict[str, value_type]
        else:
            # Nested schema - recursively build model
            return build_model("NestedSchema", value)

    # Handle string type annotations
    if isinstance(value, str):
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "any": Any,
        }
        return type_map.get(value.lower(), str)  # Default to str if unknown

    # Fallback: return as-is (already a type)
    return type(value)


def build_model(name: str, fields: Dict[str, Any]) -> Type[BaseModel]:
    """
    Build a Pydantic model from a schema dictionary.

    Supports optional fields using '?' suffix syntax:
        - "field: str"   -> required string field
        - "field?: str"  -> optional string field (can be null/omitted)

    Args:
        name: Name for the generated model
        fields: Dictionary mapping field names to type specifications

    Returns:
        Dynamically created Pydantic BaseModel class

    Example:
        >>> schema = {
        ...     "topic": "str",
        ...     "key_terms": ["str"],
        ...     "optional_field?": ["str"],  # Optional - can be null
        ...     "metrics": {"file_count": "int"}
        ... }
        >>> Model = build_model("SummarySchema", schema)
        >>> instance = Model(topic="test", key_terms=["a", "b"], metrics={"file_count": 5})
    """
    processed = {}

    for key, val in fields.items():
        # Check for optional marker (trailing ?)
        is_optional = key.endswith('?')
        clean_key = key.rstrip('?') if is_optional else key

        field_type = infer_type(val)

        if is_optional:
            # Optional field: can be None, defaults to None
            processed[clean_key] = (
                Optional[field_type],
                Field(default=None, description=f"Optional field: {clean_key}")
            )
        else:
            # Required field
            processed[clean_key] = (
                field_type,
                Field(..., description=f"Field: {clean_key}")
            )

    return create_model(name, **processed)


def validate_schema(schema: Dict[str, Any]) -> bool:
    """
    Validate that a schema dictionary is well-formed.

    Args:
        schema: Schema dictionary to validate

    Returns:
        True if valid

    Raises:
        ValueError: If schema is invalid
    """
    if not isinstance(schema, dict):
        raise ValueError(f"Schema must be a dict, got {type(schema)}")

    if len(schema) == 0:
        raise ValueError("Schema cannot be empty")

    # Try building the model to catch type errors
    try:
        build_model("ValidationTest", schema)
    except Exception as e:
        raise ValueError(f"Invalid schema: {e}")

    return True
