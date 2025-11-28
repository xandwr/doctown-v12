"""
Text chunking with embeddings.

Converts VirtualFileSystem files into semantic chunks with embeddings.
Uses simple token-based splitting with configurable overlap.
"""

import ollama
from typing import List, Optional
from .ingest.vfs import VirtualFileSystem, VFSFile
from .models import Chunk


def count_tokens(text: str) -> int:
    """
    Approximate token count for text.

    Uses simple heuristic: ~4 characters per token on average.
    This is rough but sufficient for chunking purposes.

    Args:
        text: Text to count tokens for

    Returns:
        Approximate token count
    """
    return len(text) // 4


def split_text_by_tokens(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[tuple[str, int, int]]:
    """
    Split text into chunks by approximate token count.

    Args:
        text: Full text to split
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens between consecutive chunks

    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    # Convert to lines for tracking
    lines = text.split('\n')
    chunks = []

    current_chunk = []
    current_tokens = 0
    start_line = 0

    for line_idx, line in enumerate(lines):
        line_tokens = count_tokens(line)

        # If adding this line exceeds chunk size, save current chunk
        if current_tokens + line_tokens > chunk_size and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append((chunk_text, start_line, line_idx - 1))

            # Start new chunk with overlap
            overlap_lines = []
            overlap_tokens = 0
            for prev_line in reversed(current_chunk):
                prev_tokens = count_tokens(prev_line)
                if overlap_tokens + prev_tokens > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_tokens += prev_tokens

            current_chunk = overlap_lines
            current_tokens = overlap_tokens
            start_line = line_idx - len(overlap_lines)

        current_chunk.append(line)
        current_tokens += line_tokens

    # Add final chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append((chunk_text, start_line, len(lines) - 1))

    return chunks


def get_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """
    Get embedding vector for text using Ollama.

    Args:
        text: Text to embed
        model: Ollama embedding model name

    Returns:
        Embedding vector as list of floats

    Raises:
        RuntimeError: If embedding generation fails
    """
    try:
        response = ollama.embed(model=model, input=text)
        # Ollama returns {"embeddings": [[...]]} for single input
        if "embeddings" in response and len(response["embeddings"]) > 0:
            return response["embeddings"][0]
        else:
            raise RuntimeError(f"Unexpected embedding response format: {response}")
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {e}")


class Chunker:
    """Converts VirtualFileSystem files into Chunks with embeddings."""

    def __init__(
        self,
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 50,
        embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize chunker.

        Args:
            chunk_size_tokens: Target tokens per chunk
            chunk_overlap_tokens: Overlap between consecutive chunks
            embedding_model: Ollama model for embeddings
        """
        self.chunk_size = chunk_size_tokens
        self.overlap = chunk_overlap_tokens
        self.embedding_model = embedding_model
        self._chunk_id_counter = 0

    def chunk_vfs(self, vfs: VirtualFileSystem) -> List[Chunk]:
        """
        Process all files in VFS and create chunks with embeddings.

        Args:
            vfs: Virtual file system containing source files

        Returns:
            List of Chunk objects with embeddings
        """
        chunks = []

        for path in vfs.list_files():
            vfs_file = vfs.get(path)
            if vfs_file:
                file_chunks = self._chunk_file(vfs_file)
                chunks.extend(file_chunks)

        return chunks

    def _chunk_file(self, vfs_file: VFSFile) -> List[Chunk]:
        """
        Chunk a single file.

        Args:
            vfs_file: Virtual file to chunk

        Returns:
            List of chunks for this file
        """
        # Decode file content
        try:
            text = vfs_file.data.decode('utf-8', errors='ignore')
        except Exception:
            # Skip files that can't be decoded
            return []

        # Skip empty files
        if not text.strip():
            return []

        # Split into chunks
        text_chunks = split_text_by_tokens(
            text,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

        # Create Chunk objects with embeddings
        chunks = []
        for chunk_text, start_line, end_line in text_chunks:
            # Generate embedding
            embedding = get_embedding(chunk_text, model=self.embedding_model)

            chunk = Chunk(
                chunk_id=self._chunk_id_counter,
                file_path=vfs_file.path,
                start_line=start_line + 1,  # Convert to 1-indexed
                end_line=end_line + 1,      # Convert to 1-indexed
                tokens=count_tokens(chunk_text),
                text=chunk_text,
                embedding=embedding,
                cluster_id=None  # Set during clustering
            )

            chunks.append(chunk)
            self._chunk_id_counter += 1

        return chunks


def chunk_documents(
    vfs: VirtualFileSystem,
    chunk_size_tokens: int = 500,
    chunk_overlap_tokens: int = 50,
    embedding_model: str = "nomic-embed-text"
) -> List[Chunk]:
    """
    Convenience function to chunk a VFS.

    Args:
        vfs: Virtual file system to chunk
        chunk_size_tokens: Target tokens per chunk
        chunk_overlap_tokens: Overlap between consecutive chunks
        embedding_model: Ollama embedding model

    Returns:
        List of Chunk objects with embeddings
    """
    chunker = Chunker(
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        embedding_model=embedding_model
    )
    return chunker.chunk_vfs(vfs)
