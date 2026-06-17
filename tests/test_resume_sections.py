from utils.resume_sections import (
    SECTION_SEARCH_TERMS,
    build_search_text,
    split_resume_sections,
)


def test_split_resume_sections_detects_common_headers() -> None:
    text = """
    SUMMARY
    Senior engineer with 10 years of experience.

    EXPERIENCE
    Built APIs at Example Corp.

    EDUCATION
    Master of Science in Computer Science.
    """
    sections = split_resume_sections(text)
    section_ids = [section for section, _ in sections]
    assert "summary" in section_ids
    assert "experience" in section_ids
    assert "education" in section_ids


def test_build_search_text_includes_section_topics() -> None:
    search_text = build_search_text("education", "MCA from Example University.")
    assert "Section: Education" in search_text
    assert "qualification" in search_text.lower()
    assert "MCA" in search_text


def test_education_topics_cover_qualification_queries() -> None:
    terms = SECTION_SEARCH_TERMS["education"]
    assert "qualification" in terms
    assert "maximum" in terms
