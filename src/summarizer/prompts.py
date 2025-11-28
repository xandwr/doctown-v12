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
When asked to output structured data:
- Follow the provided JSON schema exactly.
- Output only valid JSON.
- No commentary, explanation, or narration.
- No markdown.
- No trailing commas.
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
    joined = "\n\n---\n\n".join(chunks)

    # Build file paths section if provided
    paths_section = ""
    if file_paths:
        paths_list = "\n".join(f"- {p}" for p in file_paths)
        paths_section = f"""
Source files in this cluster:
{paths_list}
"""

    return f"""
Analyze this cluster of related text.

Cluster ID: {cluster_id}
Number of chunks: {len(chunks)}
{paths_section}
IMPORTANT: If you have insufficient evidence for a list field, return an empty list [].
Do not invent or hallucinate values. Empty lists are preferred over guesses.

Return findings in the exact JSON structure specified by the user.

Cluster contents:
--------------------
{joined}
--------------------

JSON schema:
{schema}

Output must be strictly valid JSON.
"""


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
    return f"""
Generate a project-level summary by synthesizing cluster summaries and metrics.

CLUSTER SUMMARIES:
--------------------
{cluster_summaries}
--------------------

{metrics}

REPRESENTATIVE SAMPLES:
--------------------
{representative_samples}
--------------------

Your task:
- Synthesize the cluster summaries into a coherent project-level view
- Incorporate the metrics where relevant
- Use representative samples for additional context
- Stay grounded in the provided information
- Output strictly according to the JSON schema below

IMPORTANT: If you have insufficient evidence for a list field, return an empty list [].
Do not invent or hallucinate values. Empty lists are preferred over guesses.

JSON schema:
{schema}

Output must be strictly valid JSON.
"""


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

