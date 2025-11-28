#!/usr/bin/env python3
"""
Test script for the summarizer module.
Verifies that Ollama is working locally and can generate structured output.
"""

from pydantic import BaseModel, Field
from typing import List
from src.summarizer.summarize import summarize_cluster


# Define a simple structured output schema
class DocumentSummary(BaseModel):
    """Expected structure for cluster summaries."""
    theme: str = Field(description="The main theme or topic")
    key_points: List[str] = Field(description="3-5 key points from the text")
    sentiment: str = Field(description="Overall sentiment: positive, neutral, or negative")


def test_basic_summarization():
    """Test basic summarization with sample text chunks."""
    print("=" * 60)
    print("Testing Ollama Summarizer")
    print("=" * 60)

    # Sample text chunks to summarize
    sample_chunks = [
        """
        Python is a high-level programming language known for its simplicity
        and readability. It has become one of the most popular languages for
        data science, machine learning, and web development.
        """,
        """
        The Python ecosystem includes powerful libraries like NumPy for
        numerical computing, pandas for data analysis, and scikit-learn
        for machine learning. These tools make Python incredibly versatile.
        """,
        """
        Python's philosophy emphasizes code readability and simplicity,
        following the principle that "there should be one obvious way to do it."
        This makes it an excellent choice for beginners and experts alike.
        """
    ]

    print("\nInput text chunks:")
    print("-" * 60)
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk.strip())

    print("\n" + "=" * 60)
    print("Calling Ollama to generate structured summary...")
    print("=" * 60)

    try:
        # Call the summarizer
        result = summarize_cluster(
            chunks_text=sample_chunks,
            cluster_id=1,
            Model=DocumentSummary,
            model_name="phi4-mini-reasoning"  # Using the default model
        )

        print("\n✓ Success! Ollama is working correctly.")
        print("\n" + "=" * 60)
        print("Structured Output:")
        print("=" * 60)
        print(f"\nTheme: {result.theme}")
        print(f"\nKey Points:")
        for i, point in enumerate(result.key_points, 1):
            print(f"  {i}. {point}")
        print(f"\nSentiment: {result.sentiment}")

        print("\n" + "=" * 60)
        print("Raw JSON:")
        print("=" * 60)
        print(result.model_dump_json(indent=2))

        return True

    except Exception as e:
        print(f"\n✗ Error occurred: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check if the model is available: ollama list")
        print("3. Pull the model if needed: ollama pull phi4-mini-reasoning")
        return False


def test_custom_schema():
    """Test with a different schema structure."""
    print("\n\n" + "=" * 60)
    print("Testing with Custom Schema")
    print("=" * 60)

    class TechnicalAnalysis(BaseModel):
        """Custom schema for technical content analysis."""
        topic: str = Field(description="Main technical topic")
        complexity: str = Field(description="beginner, intermediate, or advanced")
        concepts: List[str] = Field(description="Key technical concepts mentioned")

    sample_text = [
        """
        Docker containers provide lightweight virtualization by isolating
        applications in their own environments. Unlike virtual machines,
        containers share the host OS kernel, making them more efficient.
        """
    ]

    print("\nInput text:")
    print("-" * 60)
    print(sample_text[0].strip())

    print("\n" + "=" * 60)
    print("Calling Ollama with custom schema...")
    print("=" * 60)

    try:
        result = summarize_cluster(
            chunks_text=sample_text,
            cluster_id=2,
            Model=TechnicalAnalysis,
            model_name="phi4-mini-reasoning"
        )

        print("\n✓ Success!")
        print("\n" + "=" * 60)
        print("Structured Output:")
        print("=" * 60)
        print(f"\nTopic: {result.topic}")
        print(f"Complexity: {result.complexity}")
        print(f"Concepts: {', '.join(result.concepts)}")

        print("\n" + "=" * 60)
        print("Raw JSON:")
        print("=" * 60)
        print(result.model_dump_json(indent=2))

        return True

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n🚀 Starting Ollama Summarizer Tests\n")

    # Test 1: Basic summarization
    test1_passed = test_basic_summarization()

    # Test 2: Custom schema (only if test 1 passed)
    test2_passed = False
    if test1_passed:
        test2_passed = test_custom_schema()

    # Summary
    print("\n\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Basic Summarization: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Custom Schema:       {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Your summarizer is ready to use.")
    elif test1_passed:
        print("\n⚠️  Basic functionality works, but custom schema test failed.")
    else:
        print("\n❌ Tests failed. Check Ollama installation and configuration.")
