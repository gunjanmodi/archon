import tiktoken
from typing import List
from pydantic import BaseModel

# Get tokenizer for the same model used for embeddings
encoding = tiktoken.get_encoding("cl100k_base")


class Chunk(BaseModel):
    """Represents a text chunk with metadata for debugging and citation."""
    text: str
    index: int
    start_token: int
    end_token: int


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[Chunk]:
    """
    Split text into chunks based on token count.
    
    Args:
        text: The input text to chunk
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Number of overlapping tokens between consecutive chunks
        
    Returns:
        List of Chunk objects containing text, index, and token positions
        
    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    # Encode text to tokens
    tokens = encoding.encode(text)
    
    # If text is smaller than chunk_size, return as single chunk
    if len(tokens) <= chunk_size:
        return [Chunk(
            text=text,
            index=0,
            start_token=0,
            end_token=len(tokens)
        )]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # Define end of current chunk
        end_idx = min(start_idx + chunk_size, len(tokens))
        
        # Extract token chunk
        token_chunk = tokens[start_idx:end_idx]
        
        # Decode tokens back to text
        c = Chunk(
            text=encoding.decode(token_chunk),
            index=len(chunks),
            start_token=start_idx,
            end_token=end_idx
        )
        chunks.append(c)
        
        # Move start index forward by (chunk_size - overlap)
        start_idx += chunk_size - chunk_overlap
    
    return chunks

