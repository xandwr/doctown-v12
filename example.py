#!/usr/bin/env python3
"""
Example usage of the doctown pipeline.

This demonstrates the full workflow:
1. Ingest a codebase (GitHub or ZIP)
2. Apply an intent spec (YAML)
3. Generate structured outputs
4. Save as .docpack
5. Load and inspect the .docpack
"""

from pathlib import Path
from src.pipeline import run_pipeline, quick_run
from src.docpack import load_docpack
import json


def example_basic():
    """Basic example using quick_run helper."""
    print("=== BASIC EXAMPLE ===\n")

    # Run with built-in minimal intent
    docpack = quick_run(
        source="https://github.com/anthropics/anthropic-sdk-python",
        intent="minimal",
        output="example_output/anthropic-sdk.docpack"
    )

    print("\n📦 Generated docpack:")
    print(f"  - Files: {docpack.manifest.file_count}")
    print(f"  - Chunks: {docpack.manifest.chunk_count}")
    print(f"  - Clusters: {docpack.manifest.cluster_count}")
    print(f"  - Project summary fields: {list(docpack.project_summary.keys())}")


def example_custom_intent():
    """Example with custom intent and configuration."""
    print("\n=== CUSTOM INTENT EXAMPLE ===\n")

    from src.models import PipelineConfig

    config = PipelineConfig(
        chunk_size_tokens=300,      # Smaller chunks
        num_clusters=5,             # Fewer clusters
        embedding_model="nomic-embed-text",
        summarization_model="phi4-mini-reasoning"
    )

    docpack = run_pipeline(
        source="https://github.com/user/small-repo",
        intent_yaml="src/intent/examples/codebase_docs.yaml",
        output_path="example_output/custom.docpack",
        config=config
    )

    print(f"\n✓ Created custom docpack with {docpack.manifest.cluster_count} clusters")


def example_inspect_docpack():
    """Example of loading and inspecting a .docpack."""
    print("\n=== INSPECT DOCPACK EXAMPLE ===\n")

    # Load existing docpack
    docpack = load_docpack("example_output/anthropic-sdk.docpack")

    print(f"Loaded docpack: {docpack.manifest.intent_name}")
    print(f"Created: {docpack.manifest.created_at}")
    print(f"Source: {docpack.manifest.source_identifier}\n")

    # Inspect clusters
    print("Clusters:")
    for cluster in docpack.clusters[:3]:  # First 3 clusters
        print(f"\n  Cluster {cluster.cluster_id}:")
        print(f"    Chunks: {len(cluster.chunk_ids)}")
        if cluster.summary:
            print(f"    Summary fields: {list(cluster.summary.keys())}")
            # Print topic if it exists
            if "topic" in cluster.summary:
                print(f"    Topic: {cluster.summary['topic']}")

    # Inspect project summary
    print("\n\nProject Summary:")
    for key, value in docpack.project_summary.items():
        if isinstance(value, str) and len(value) < 100:
            print(f"  {key}: {value}")
        elif isinstance(value, list) and len(value) < 5:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: <{type(value).__name__}>")

    # Export project summary to JSON
    output_file = "example_output/project_summary.json"
    with open(output_file, 'w') as f:
        json.dump(docpack.project_summary, f, indent=2)
    print(f"\n✓ Exported project summary to: {output_file}")


def example_local_zip():
    """Example using a local ZIP file."""
    print("\n=== LOCAL ZIP EXAMPLE ===\n")

    # First create a test ZIP (you'd normally have this already)
    import zipfile
    import io

    test_zip = "example_output/test_code.zip"
    Path("example_output").mkdir(exist_ok=True)

    with zipfile.ZipFile(test_zip, 'w') as zf:
        zf.writestr("main.py", "def hello():\n    print('Hello world')\n")
        zf.writestr("utils.py", "def add(a, b):\n    return a + b\n")
        zf.writestr("README.md", "# Test Project\n\nA simple test project.\n")

    print(f"Created test ZIP: {test_zip}")

    # Run pipeline on local ZIP
    docpack = quick_run(
        source=test_zip,
        intent="minimal",
        output="example_output/test_code.docpack"
    )

    print(f"\n✓ Processed local ZIP")
    print(f"  Files: {docpack.manifest.file_count}")
    print(f"  Chunks: {docpack.manifest.chunk_count}")


def example_programmatic():
    """Example of programmatic usage without CLI."""
    print("\n=== PROGRAMMATIC EXAMPLE ===\n")

    from src.ingest.ingest import ingest_source
    from src.chunking import chunk_documents
    from src.clustering import cluster_documents
    from src.intent.spec import IntentSpec
    from src.intent.orchestrator import apply_intent
    from src.models import DocpackManifest

    # 1. Ingest
    print("Step 1: Ingesting...")
    vfs = ingest_source("https://github.com/user/tiny-repo")
    print(f"  Loaded {len(vfs.list_files())} files")

    # 2. Chunk
    print("Step 2: Chunking...")
    chunks = chunk_documents(vfs, chunk_size_tokens=400)
    print(f"  Created {len(chunks)} chunks")

    # 3. Cluster
    print("Step 3: Clustering...")
    clusters = cluster_documents(chunks, num_clusters=3)
    print(f"  Formed {len(clusters)} clusters")

    # 4. Load intent
    print("Step 4: Loading intent...")
    intent = IntentSpec.from_yaml("src/intent/examples/minimal.yaml")

    # 5. Create manifest
    manifest = DocpackManifest(
        source_type="github",
        source_identifier="https://github.com/user/tiny-repo",
        intent_name=intent.name,
        intent_description=intent.description,
        file_count=len(vfs.list_files()),
        chunk_count=len(chunks),
        cluster_count=len(clusters),
        total_tokens=sum(c.tokens for c in chunks),
        embedding_dim=len(chunks[0].embedding) if chunks else 0
    )

    # 6. Apply intent
    print("Step 5: Applying intent...")
    docpack = apply_intent(intent, chunks, clusters, manifest)

    print("\n✓ Programmatically created docpack")
    print(f"  Clusters with summaries: {sum(1 for c in docpack.clusters if c.summary)}")
    print(f"  Project summary fields: {len(docpack.project_summary)}")


if __name__ == "__main__":
    import sys

    # Create output directory
    Path("example_output").mkdir(exist_ok=True)

    print("Doctown Pipeline Examples")
    print("=" * 50)

    # Run examples
    try:
        # Start with local ZIP (doesn't require internet)
        example_local_zip()

        # Show how to inspect
        example_inspect_docpack()

        # Uncomment to run internet-based examples:
        # example_basic()
        # example_custom_intent()
        # example_programmatic()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
