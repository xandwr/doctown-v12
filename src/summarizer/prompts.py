# prompts.py
# system + cluster + file summary prompts

# Example:
# User defines:
# {
#   "theme": "string",
#   "key_points": ["string"],
#   "risk_level": "string"
# }
# The summarizer will fill it in.

"""
Prompts for semantic analysis, clustering interpretation,
and structured summarization for arbitrary document corpora.

These prompts avoid domain assumptions and work for:
- technical documents
- business or legal texts
- knowledge bases
- research papers
- personal notes
- arbitrary mixed-format ingestion

They are also compatible with structure specifications:
the model can output structured JSON following a schema
defined dynamically by the user.

All prompts are model-agnostic and contain no special tags.
"""

# =====================================================================
# SYSTEM PROMPTS (context setters)
# =====================================================================

SYSTEM_ANALYZER = """
You are a semantic analysis model.
Your job is to examine groups of related text and produce grounded,
factually faithful interpretations without inventing details.

Core rules:
- Stay strictly grounded in the given text.
- Identify themes, patterns, and relationships.
- Avoid domain assumptions unless information is explicit.
- No hallucination or speculation.
- Use the user's structure specification when provided.
"""

SYSTEM_SUMMARIZER = """
You summarize documents or clusters of documents
into clear, factual, domain-neutral descriptions.

Guidelines:
- Focus on meaning, intent, and relationships.
- Avoid technical jargon unless the input uses it.
- No markdown or formatting unless requested.
- If a structure spec is provided, output strictly in that format.
"""

SYSTEM_STRUCTURED_OUTPUT = """
You are a code/document analyzer that extracts factual information into structured JSON.

CRITICAL RULES:
- Extract REAL identifiers, names, and text from the input - never invent or hallucinate
- Output ONLY valid JSON matching the schema - no explanation, markdown, or commentary
- Use empty lists [] or null for fields with insufficient evidence
- NEVER echo task instructions back as field values
- File paths are strong hints for module/package names
"""


# =====================================================================
# USER PROMPTS (generation tasks)
# =====================================================================

def make_item_summary_prompt(text: str) -> str:
    return f"""
Analyze the following document or text segment.

Your task:
- Extract the key ideas, themes, relationships, and purposes.
- Avoid adding meaning not present in the text.
- Present the distilled meaning concisely.

Text:
--------------------
{text}
--------------------
"""


def make_cluster_summary_prompt(texts: list[str]) -> str:
    joined = "\n\n---\n\n".join(texts)
    return f"""
You are analyzing a group of semantically related text segments.

Your goal:
- Identify the central theme(s)
- Explain what the texts collectively represent
- Highlight any connecting patterns or shared purpose
- Remain fully grounded in the provided content

Cluster:
--------------------
{joined}
--------------------
"""


def make_project_summary_prompt(representative_texts: list[str]) -> str:
    joined = "\n\n---\n\n".join(representative_texts)
    return f"""
You will generate a high-level overview of the entire corpus
based solely on representative samples.

Describe:
- The overarching purpose of the corpus
- The main themes present
- The types of information contained
- Any relationships or structure you can infer
- Avoid all speculation beyond what is clearly supported

Do NOT use markdown.
Do NOT hallucinate details.
Keep it general and factual.

Corpus excerpts:
--------------------
{joined}
--------------------
"""


# =====================================================================
# STRUCTURE-SPECIFIC PROMPTS
# =====================================================================

def make_structured_output_prompt(text: str, schema: dict) -> str:
    """
    User provides a Pydantic or JSON-schema-like structure.
    The model must fill it.

    This wraps SYSTEM_STRUCTURED_OUTPUT automatically
    when passed to the model as system+user messages.
    """
    return f"""
You must analyze the following text and produce output that matches
the provided JSON schema exactly. No additional fields may be added.

Text:
--------------------
{text}
--------------------

JSON schema:
{schema}

Output only valid JSON following the schema.
"""


def make_structured_cluster_prompt(
    cluster_id: int,
    chunks: list[str],
    schema: dict,
    file_paths: list[str] | None = None
) -> str:
    """
    Generate a prompt for structured cluster summarization.

    Args:
        cluster_id: ID of the cluster being summarized
        chunks: List of text chunks in this cluster
        schema: User-defined schema for output
        file_paths: Optional list of source file paths in this cluster

    Returns:
        Formatted prompt string
    """
    # Build file paths section - CRITICAL for grounding small models
    paths_section = ""
    if file_paths:
        paths_list = "\n".join(f"  • {p}" for p in sorted(file_paths))
        paths_section = f"""
═══ SOURCE FILES ═══
{paths_list}

"""

    # Format chunks with clear visual boundaries
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append(f"── Chunk {i+1} ──\n{chunk}")
    joined = "\n\n".join(formatted_chunks)

    return f"""Extract information from this code/documentation cluster.

{paths_section}═══ CONTENT ({len(chunks)} chunks) ═══

{joined}

═══ END CONTENT ═══

EXTRACTION TASK:
Analyze the content above and fill the JSON schema. Follow these rules:

1. EXTRACT REAL DATA: Copy actual function names, class names, imports, and identifiers verbatim from the code
2. USE FILE PATHS: Infer module/package names from paths (e.g., "src/auth/login.py" → module is "auth.login")
3. FIND DEPENDENCIES: List real imports you see (e.g., "import pandas", "from fastapi import", "use tokio::")
4. SUMMARIZE FROM TEXT: For description fields, paraphrase what the code does based on comments, docstrings, or logic
5. EMPTY IS OK: Use [] for lists and null for optional fields if no evidence exists
6. DO NOT ECHO INSTRUCTIONS: Never put task instructions or schema field names as values

SCHEMA TO FILL:
{schema}

Respond with valid JSON only."""


def make_structured_project_prompt(
    cluster_summaries: str,
    metrics: str,
    representative_samples: str,
    schema: dict
) -> str:
    """
    Generate a prompt for structured project-level summarization.

    This is the deterministic merge: project_summary = merge(clusters + metrics + intent)

    Args:
        cluster_summaries: All cluster summaries combined
        metrics: Dataset metrics (file count, token count, etc.)
        representative_samples: Sample chunks from each cluster
        schema: User-defined schema for output

    Returns:
        Formatted prompt string
    """
    return f"""Synthesize a project-level summary from these cluster analyses.

═══ CLUSTER SUMMARIES ═══
{cluster_summaries}

═══ METRICS ═══
{metrics}

═══ SAMPLE CODE/TEXT ═══
{representative_samples}

═══ END INPUT ═══

SYNTHESIS TASK:
Create a unified project summary by combining insights from all clusters above.

RULES:
1. AGGREGATE: Combine module names, functions, and patterns found across clusters
2. IDENTIFY THEMES: What is this project/codebase about? What problem does it solve?
3. LIST REAL ARTIFACTS: Only include module names, entry points, and tech stack items that appear in the cluster data
4. USE METRICS: Reference actual file counts, token counts from the metrics section
5. EMPTY IS OK: Use [] for lists and null for optional fields if no clear evidence exists
6. BE SPECIFIC: "Rust CLI tool for document analysis" is better than "A software project"

SCHEMA TO FILL:
{schema}

Respond with valid JSON only."""


# =====================================================================
# JSON INSTRUCTIONS (shared across tasks)
# =====================================================================

JSON_INSTRUCTIONS = """
- Output ONLY valid JSON.
- No trailing commas.
- No comments or explanation.
- Do not wrap JSON in markdown.
- Do not add or remove fields from the schema.
"""

