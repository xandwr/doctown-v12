"""
CLI entry point for doctown.

Usage:
    python -m src <source> --intent <intent_yaml> --output <output_path>

Examples:
    # Use built-in intent
    python -m src https://github.com/user/repo --intent minimal

    # Custom intent YAML
    python -m src my_project.zip --intent custom/my_intent.yaml --output results.docpack

    # Specify all parameters
    python -m src https://github.com/user/repo \\
        --intent minimal \\
        --output repo.docpack \\
        --chunks 300 \\
        --clusters 5 \\
        --embedding nomic-embed-text \\
        --model phi4-mini-reasoning
"""

import argparse
import sys
from pathlib import Path

from .pipeline import run_pipeline
from .models import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Doctown: Intent-driven document analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "source",
        help="Source to ingest (GitHub URL, ZIP file, or directory)"
    )

    # Intent specification
    parser.add_argument(
        "--intent", "-i",
        default="minimal",
        help="Intent YAML file or built-in intent name (default: minimal)"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output .docpack path (default: auto-generated)"
    )

    # Pipeline parameters
    parser.add_argument(
        "--chunks",
        type=int,
        default=500,
        help="Chunk size in tokens (default: 500)"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Chunk overlap in tokens (default: 50)"
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=10,
        help="Number of clusters (default: 10)"
    )

    parser.add_argument(
        "--clustering",
        choices=["kmeans", "hierarchical"],
        default="kmeans",
        help="Clustering method (default: kmeans)"
    )

    parser.add_argument(
        "--embedding",
        default="nomic-embed-text",
        help="Ollama embedding model (default: nomic-embed-text)"
    )

    parser.add_argument(
        "--model",
        default="phi4-mini-reasoning",
        help="Ollama summarization model (default: phi4-mini-reasoning)"
    )

    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Exclude raw files from .docpack"
    )

    args = parser.parse_args()

    # Resolve intent path
    intent_path = Path(args.intent)
    if not intent_path.exists():
        # Try built-in intents
        builtin_path = Path(__file__).parent / "intent" / "examples" / f"{args.intent}.yaml"
        if builtin_path.exists():
            intent_path = builtin_path
        else:
            print(f"Error: Intent file not found: {args.intent}", file=sys.stderr)
            print(f"  Tried: {args.intent}", file=sys.stderr)
            print(f"  Tried: {builtin_path}", file=sys.stderr)
            sys.exit(1)

    # Auto-generate output path if not provided
    output_path = args.output
    if output_path is None:
        if args.source.startswith("http"):
            repo_name = args.source.rstrip('/').split('/')[-1]
        else:
            repo_name = Path(args.source).stem
        output_path = f"{repo_name}.docpack"

    # Build config
    config = PipelineConfig(
        chunk_size_tokens=args.chunks,
        chunk_overlap_tokens=args.overlap,
        num_clusters=args.clusters,
        clustering_method=args.clustering,
        embedding_model=args.embedding,
        summarization_model=args.model,
        include_raw_files=not args.no_raw
    )

    # Run pipeline
    try:
        docpack = run_pipeline(
            source=args.source,
            intent_yaml=intent_path,
            output_path=output_path,
            config=config,
            include_raw=not args.no_raw
        )
        print(f"\n✓ Success! Docpack saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
