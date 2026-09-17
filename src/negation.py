import re
from .utils import sentences

NEGATION = re.compile(
    r"\b(?:no|without|never|neither|denied|absence of|not associated with|"
    r"did not|does not|was not|were not|is not|not observed|not reported|not performed|"
    r"kein(?:e|en|er|es|em)?|nicht|ohne|nie|niemals|weder|verneint)\b", re.I
)


def assess_negation(text: str, concepts: str = "") -> tuple[list[str], bool]:
    """Clause-level heuristic. Mixed polarity is deliberately not affirmative evidence."""
    negative, positive = [], []
    for sentence in sentences(text):
        for clause in re.split(r"\b(?:but|however|whereas|although|aber|jedoch|allerdings|sondern|obwohl)\b|;", sentence, flags=re.I):
            if concepts and not re.search(concepts, clause, re.I):
                continue
            # 'Not only' is a pseudo-negation.
            candidate = re.sub(r"\b(?:not only|nicht nur|nicht ausgeschlossen|nicht auszuschließen|"
                               r"kann nicht ausgeschlossen werden|cannot be ruled out)\b", "", clause, flags=re.I)
            if NEGATION.search(candidate):
                negative.append(clause.strip())
            else:
                positive.append(clause.strip())
    return negative, bool(negative and positive)
