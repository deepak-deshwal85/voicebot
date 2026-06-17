from __future__ import annotations

import re
from typing import Literal

KnowledgeSource = Literal["website", "pdf", "both"]

_WEBSITE_TERMS = {
    "account",
    "adviser",
    "advisor",
    "annuity",
    "bond",
    "bonds",
    "company",
    "etf",
    "fee",
    "fees",
    "financial",
    "fund",
    "funds",
    "invest",
    "investment",
    "investments",
    "investor",
    "isa",
    "market",
    "mortgage",
    "pension",
    "pensions",
    "portfolio",
    "product",
    "products",
    "retirement",
    "savings",
    "service",
    "services",
    "share",
    "shares",
    "sipp",
    "stock",
    "stocks",
    "tax",
    "trading",
    "transfer",
    "wealth",
}

_PDF_TERMS = {
    "biography",
    "certification",
    "college",
    "cv",
    "degree",
    "education",
    "employer",
    "employment",
    "experience",
    "job",
    "personal",
    "profile",
    "project",
    "projects",
    "qualification",
    "resume",
    "school",
    "skill",
    "skills",
    "university",
    "work",
}


def route_knowledge_source(query: str) -> KnowledgeSource:
    words = set(re.findall(r"[a-z]+", query.lower()))
    if not words:
        return "both"

    website_score = len(words & _WEBSITE_TERMS)
    pdf_score = len(words & _PDF_TERMS)

    if website_score > pdf_score and website_score > 0:
        return "website"
    if pdf_score > website_score and pdf_score > 0:
        return "pdf"
    if website_score > 0 and pdf_score > 0:
        return "both"
    return "both"
