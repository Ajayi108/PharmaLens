from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    document_name: str
    page_number: int
    section: str
    text: str
    chunk_id: str
    normalized_section: str = "Unknown"
    language: str = "und"
    document_id: str = ""


@dataclass
class ScreeningCriterion:
    id: str
    name: str
    queries: list[str]
    question: str
    concepts: str = ""
    question_de: str = ""


@dataclass
class RetrievedEvidence:
    criterion: str
    text: str
    document_name: str
    page_number: int
    section: str
    similarity_score: float
    negated: bool
    chunk_id: str
    negative_sentences: list[str] = field(default_factory=list)
    mixed: bool = False
    normalized_section: str = "Unknown"
    language: str = "und"
    document_id: str = ""


@dataclass
class ScreeningResult:
    criterion: ScreeningCriterion
    result: str
    evidence: list[RetrievedEvidence]


@dataclass
class ArticleScreeningResult:
    document_name: str
    results: list[ScreeningResult]
    status: str
    explanation: str
