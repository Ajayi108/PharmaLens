import re

# Small, explicit dictionaries: extension point for a biomedical NER provider.
PATTERNS = {
    "Drug": r"\b(?:pembrolizumab|nivolumab|ipilimumab|aspirin|acetylsalicylsäure|ibuprofen|paracetamol|acetaminophen|methotrexat(?:e)?|warfarin|amoxicillin|cisplatin|predniso(?:ne|n|lon)|metformin)\b",
    "Adverse event": r"\b(?:colitis|kolitis|hepatitis|rash|hautausschlag|hautreaktion|leberschädigung|liver injury|nausea|übelkeit|vomiting|erbrechen|neutropenia|neutropenie|anaphylaxis|anaphylaxie|diarrh(?:ea|oea|ö)|durchfall|thrombocytopenia|thrombozytopenie|pneumonitis|toxicity|toxizität)\b",
    "Age": r"\b\d{1,3}[ -](?:year|month|day)s?[ -]old\b|\baged?\s+\d{1,3}\b|\b\d{1,3}[ -]jährig(?:e|er|en|es)?\b|\b\d{1,3}\s+Jahre\s+alt\b",
    "Sex": r"\b(?:male|female|man|woman|boy|girl|mann|frau|junge|mädchen|männlich(?:e|er|en|es)?|weiblich(?:e|er|en|es)?)\b",
    "Dosage": r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|mL|IU|IE)(?:\s*/\s*(?:kg|day|Tag|m2|m²))?\b",
    "Date": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4}\b",
    "Medical condition": r"\b(?:melanoma|melanom|cancer|krebs|diabetes|hypertension|hypertonie|arthritis|lymphoma|lymphom|asthma|leukemia|leukämie)\b",
}


def detect_entities(text: str) -> list[dict]:
    return [dict(entity=match.group(), type=kind, start=match.start(), end=match.end())
            for kind, pattern in PATTERNS.items() for match in re.finditer(pattern, text, re.I)]
