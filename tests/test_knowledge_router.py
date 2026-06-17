from utils.knowledge_router import route_knowledge_source


def test_routes_financial_questions_to_website() -> None:
    assert (
        route_knowledge_source("Tell me about pension transfer services") == "website"
    )
    assert route_knowledge_source("What ISA products do you offer?") == "website"


def test_routes_document_questions_to_pdf() -> None:
    assert route_knowledge_source("What skills are listed in the resume?") == "pdf"
    assert route_knowledge_source("Tell me about their work experience") == "pdf"


def test_routes_ambiguous_questions_to_both() -> None:
    assert route_knowledge_source("Tell me more") == "both"
