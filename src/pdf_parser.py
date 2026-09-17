import re
import pymupdf as fitz
from .language import detect_language

SECTION_NAMES = {
    "Abstract": ["Abstract", "Zusammenfassung"],
    "Introduction": ["Introduction", "Einleitung"],
    "Background": ["Background", "Hintergrund"],
    "Methods": ["Methods", "Methoden", "Materials and Methods", "Patients and Methods", "Material und Methoden", "Materialien und Methoden"],
    "Results": ["Results", "Ergebnisse"],
    "Discussion": ["Discussion", "Diskussion"],
    "Conclusion": ["Conclusion", "Conclusions", "Schlussfolgerung", "Schlussfolgerungen", "Fazit"],
    "Case Presentation": ["Case Presentation", "Case Report", "Fallbericht", "Fallbeschreibung"],
    "Safety": ["Safety", "Sicherheit"],
    "Adverse Events": ["Adverse Events", "Nebenwirkungen", "Unerwünschte Ereignisse"],
    "References": ["References", "Bibliography", "Literatur", "Literaturverzeichnis"],
}
SECTION_LOOKUP = {heading.casefold(): canonical for canonical, headings in SECTION_NAMES.items() for heading in headings}
HEADINGS = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(" +
                      "|".join(re.escape(name) for name in SECTION_LOOKUP) + r")\s*:?$", re.I)


def normalize_section(heading: str) -> str:
    match = HEADINGS.fullmatch(heading.strip())
    return SECTION_LOOKUP[match.group(1).casefold()] if match else "Unknown"


def extract_pdf(data: bytes, filename: str) -> tuple[list[dict], int, list[int]]:
    """Return section blocks with page provenance; never join across pages."""
    blocks, empty_pages = [], []
    section = "Unknown"
    with fitz.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDF. Upload an unlocked copy.")
        page_count = len(document)
        for page_number, page in enumerate(document, 1):
            original = page.get_text("text", sort=True)
            if not original.strip():
                empty_pages.append(page_number)
                continue
            buffer = []

            def flush():
                if buffer:
                    blocks.append(dict(document=filename, page=page_number,
                                       section=section, normalized_section=normalize_section(section),
                                       original_text="\n".join(buffer)))
                    buffer.clear()

            for line in original.splitlines():
                match = HEADINGS.fullmatch(line.strip())
                if match:
                    flush()
                    section = line.strip()
                else:
                    buffer.append(line)
            flush()
    language = detect_language("\n".join(block["original_text"] for block in blocks))
    for block in blocks:
        block["language"] = language
    return blocks, page_count, empty_pages
