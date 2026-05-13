from citations import extract_citations, find_fabricated_citations


def test_extract_citations_with_single_chunk_reference_returns_chunk_number():
    text = "Paris is the capital of France [Chunk 2]."

    assert extract_citations(text) == {2}


def test_extract_citations_with_multiple_chunk_references_returns_all_chunk_numbers():
    text = "The answer is supported by both sources [Chunk 1, Chunk 3]."

    assert extract_citations(text) == {1, 3}


def test_extract_citations_with_no_chunk_references_returns_empty_set():
    text = "The provided context does not contain enough information to answer this reliably."

    assert extract_citations(text) == set()


def test_find_fabricated_citations_flags_chunk_numbers_outside_retrieved_range():
    text = "Paris is the capital of France [Chunk 5]."

    assert find_fabricated_citations(text, chunk_count=2) == {5}
