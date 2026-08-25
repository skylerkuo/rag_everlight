from rag_app.preprocess.pdf_to_md import _context_page_indices, _page_role_label


def test_context_radius_one_middle_page():
    assert _context_page_indices(2, 5, 1) == [1, 2, 3]


def test_context_radius_one_document_edges():
    assert _context_page_indices(0, 5, 1) == [0, 1]
    assert _context_page_indices(4, 5, 1) == [3, 4]


def test_context_radius_zero_is_original_single_page_mode():
    assert _context_page_indices(2, 5, 0) == [2]


def test_page_labels_identify_target_and_context():
    assert "PREVIOUS CONTEXT PAGE" in _page_role_label(1, 2)
    assert "TARGET PAGE" in _page_role_label(2, 2)
    assert "NEXT CONTEXT PAGE" in _page_role_label(3, 2)
