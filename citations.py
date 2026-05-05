import re
from typing import Set


CITATION_PATTERN = re.compile(r"\[(Chunk\s+\d+(?:\s*,\s*Chunk\s+\d+)*)\]")
CHUNK_NUMBER_PATTERN = re.compile(r"Chunk\s+(\d+)")


def extract_citations(text: str) -> Set[int]:
    """
    Extract cited chunk numbers from text.

    Supports citations like [Chunk 2] and [Chunk 1, Chunk 3].
    """
    citations = set()

    for citation_group in CITATION_PATTERN.findall(text):
        for chunk_number in CHUNK_NUMBER_PATTERN.findall(citation_group):
            citations.add(int(chunk_number))

    return citations


def find_fabricated_citations(text: str, chunk_count: int) -> Set[int]:
    """
    Return cited chunk numbers that are outside the retrieved chunk range.
    """
    return {
        citation
        for citation in extract_citations(text)
        if citation < 1 or citation > chunk_count
    }
