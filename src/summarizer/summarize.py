# summarize.py

from __future__ import annotations

from typing import List, Type
from ollama import chat
from pydantic import BaseModel, ValidationError

from .prompts import (
    SYSTEM_SUMMARIZER,
    JSON_INSTRUCTIONS,
    make_cluster_summary_prompt,
    make_structured_cluster_prompt,
)


def call_llm(model: str, system: str, user: str, schema: dict) -> str:
    """
    Core structured-output call.

    - Fully JSON deterministic
    - Compatible with any Ollama model that supports 'format='
    """
    response = chat(
        model=model,
        format=schema,                  # enforce EXACT JSON structure
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
            {"role": "user",   "content": JSON_INSTRUCTIONS},
        ]
    )

    if response.message is None or response.message.content is None:
        raise RuntimeError("LLM returned no content")

    return response.message.content


def summarize_cluster(
    chunks_text: List[str],
    cluster_id: int,
    Model: Type[BaseModel],
    model_name: str = "phi4-mini-reasoning",
) -> BaseModel:
    """
    Summarize a semantic cluster using a user-specified Pydantic model
    defining the exact JSON structure.

    Params:
        chunks_text : List[str]  → the raw text of the cluster's chunks
        cluster_id  : int        → for context only
        Model       : BaseModel  → pydantic model to structure output
        model_name  : str        → Ollama model to use
    """
    # Build the system+user prompt text
    schema = Model.model_json_schema()

    prompt = make_structured_cluster_prompt(
        texts=chunks_text,
        schema=schema,
    )

    # Perform LLM call
    result_json = call_llm(
        model=model_name,
        system=SYSTEM_SUMMARIZER,
        user=prompt,
        schema=schema,
    )

    # Validate JSON
    try:
        return Model.model_validate_json(result_json)
    except ValidationError as e:
        raise RuntimeError(
            f"Invalid structured JSON returned by model:\n{e}\nJSON was:\n{result_json}"
        )
