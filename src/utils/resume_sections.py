from __future__ import annotations

import re

SECTION_ORDER = (
    "summary",
    "experience",
    "education",
    "skills",
    "certifications",
    "projects",
)

SECTION_LABELS: dict[str, str] = {
    "summary": "Summary",
    "experience": "Work Experience",
    "education": "Education",
    "skills": "Skills",
    "certifications": "Certifications",
    "projects": "Projects",
    "general": "Resume",
}

# Query terms that should prefer a section at search time (data-agnostic).
SECTION_SEARCH_TERMS: dict[str, frozenset[str]] = {
    "summary": frozenset(
        {
            "summary",
            "overview",
            "profile",
            "about",
            "background",
            "designation",
            "current",
            "present",
            "title",
            "role",
        }
    ),
    "experience": frozenset(
        {
            "experience",
            "employment",
            "work",
            "job",
            "company",
            "employer",
            "designation",
            "role",
            "position",
            "title",
            "currently",
            "present",
            "responsibilities",
            "career",
        }
    ),
    "education": frozenset(
        {
            "education",
            "qualification",
            "qualifications",
            "degree",
            "degrees",
            "university",
            "college",
            "school",
            "academic",
            "postgraduate",
            "undergraduate",
            "graduate",
            "masters",
            "master",
            "bachelor",
            "phd",
            "diploma",
            "highest",
            "maximum",
        }
    ),
    "skills": frozenset(
        {
            "skills",
            "skill",
            "technologies",
            "technical",
            "expertise",
            "competencies",
            "proficient",
            "stack",
            "tools",
            "programming",
            "languages",
        }
    ),
    "certifications": frozenset(
        {
            "certification",
            "certifications",
            "certificate",
            "certificates",
            "certified",
            "license",
            "licenses",
            "accreditation",
        }
    ),
    "projects": frozenset(
        {
            "project",
            "projects",
            "portfolio",
            "built",
            "developed",
        }
    ),
}

_SECTION_HEADER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:PROFESSIONAL\s+)?SUMMARY\b", re.I), "summary"),
    (re.compile(r"\b(?:PROFILE|OBJECTIVE)\b", re.I), "summary"),
    (
        re.compile(
            r"\b(?:WORK\s+)?EXPERIENCE|EMPLOYMENT(?:\s+HISTORY)?|PROFESSIONAL\s+EXPERIENCE\b",
            re.I,
        ),
        "experience",
    ),
    (re.compile(r"\bEDUCATION\b", re.I), "education"),
    (
        re.compile(r"\b(?:TECHNICAL\s+)?SKILLS|CORE\s+COMPETENCIES\b", re.I),
        "skills",
    ),
    (re.compile(r"\bCERTIFICATIONS?\b", re.I), "certifications"),
    (re.compile(r"\bPROJECTS?\b", re.I), "projects"),
]


def section_keywords(section: str) -> list[str]:
    return sorted(SECTION_SEARCH_TERMS.get(section, frozenset()))


def split_resume_sections(text: str) -> list[tuple[str, str]]:
    """Split resume text into (section_id, body) segments using common headers."""
    matches: list[tuple[int, str]] = []
    for pattern, section_id in _SECTION_HEADER_PATTERNS:
        for match in pattern.finditer(text):
            matches.append((match.start(), section_id))

    if not matches:
        cleaned = text.strip()
        return [("general", cleaned)] if cleaned else []

    matches.sort(key=lambda item: item[0])
    deduped: list[tuple[int, str]] = []
    last_pos = -1
    for pos, section_id in matches:
        if pos == last_pos:
            continue
        if deduped and deduped[-1][0] == pos:
            continue
        deduped.append((pos, section_id))
        last_pos = pos

    sections: list[tuple[str, str]] = []
    for index, (pos, section_id) in enumerate(deduped):
        end = deduped[index + 1][0] if index + 1 < len(deduped) else len(text)
        body = text[pos:end].strip()
        if body:
            sections.append((section_id, body))
    return sections


def build_search_text(section: str, text: str) -> str:
    label = SECTION_LABELS.get(section, section.title())
    keywords = ", ".join(section_keywords(section))
    return f"Section: {label}\nTopics: {keywords}\n{text}"
