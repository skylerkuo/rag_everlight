from rag_app.chunking.markdown_chunker import ApproxTokenCounter, Section, chunk_section, split_sections


def test_heading_sections():
    md = "# Product\nIntro text.\n\n## Features\n- Fast\n- Safe\n\n## Applications\nFactory use."
    sections = split_sections(md)
    assert len(sections) == 3
    assert sections[1].heading_path == ["Product", "Features"]


def test_chunks_do_not_cross_heading():
    counter = ApproxTokenCounter()
    s = Section(["A"], "Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    chunks = chunk_section(s, counter, target_tokens=5, max_tokens=10, overlap_tokens=2)
    assert chunks
    assert all(counter.count(x) <= 20 for x in chunks)
