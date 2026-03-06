from difflib import SequenceMatcher
import re

def normalize_class_name(class_input):
    """
    Converts AI-extracted input to SQL-friendly pattern
    Handles: KG, 1, 2, 3, etc.
    """
    class_input = class_input.strip().upper()

    # Handle KG classes
    if class_input.startswith("KG"):
        # Accept "KG1", "KG-1", "KG 1"
        return f"%KG%{class_input[-1]}%"

    # Handle numeric classes 1-12
    if class_input.isdigit():
        return f"%{class_input}%"

        # Handle 1st, 2nd, 3rd, 4th classes
    mapping = {
        "1ST": "1", "2ND": "2", "3RD": "3", "4TH": "4", "5TH": "5",
        "6TH": "6", "7TH": "7", "8TH": "8", "9TH": "9", "10TH": "10",
        "11TH": "11", "12TH": "12"
    }
    for k, v in mapping.items():
        if k in class_input:
            return f"%{v}%"

    # Default fallback
    return f"%{class_input}%"


def exact_first_name_match(source: str, student_name: str, threshold: float = 0.7) -> bool:
    """
    Checks if the first name in 'source' is fuzzy-similar to 'student_name'.
    Example: source="Rohan Verma", student_name="vohan" -> True
    """
    if not source or not student_name:
        return False

    # 1. Normalize and extract the first word (First Name)
    source_first_name = source.split()[0].lower()
    target_first_name = student_name.split()[0].lower()

    # 2. Check for exact match first (Performance)
    if source_first_name == target_first_name:
        return True

    # 3. Fuzzy ratio (handles Vohan vs Rohan)
    similarity = SequenceMatcher(None, source_first_name, target_first_name).ratio()

    return similarity >= threshold





# ==============================================================
# SEARCH CONTENT CLEANER
# ==============================================================

def _clean_content(text: str) -> str:
    """
    Strip everything that makes TTS sound bad:
    - Unicode / IPA phonetic symbols
    - Parenthetical asides like ([pronunciation]; born ...)
    - URLs
    - Markdown formatting
    - Extra whitespace
    Truncate at a sentence boundary around 250 chars.
    """
    if not text:
        return ""

    # Decode escaped unicode (e.g. \u0259) then drop all non-ASCII
    try:
        text = text.encode("utf-8").decode("unicode_escape")
    except Exception:
        pass
    text = text.encode("ascii", errors="ignore").decode("ascii")

    # Remove parenthetical content (IPA, dates, clarifications)
    text = re.sub(r'\([^)]{0,120}\)', ' ', text)
    text = re.sub(r'\[[^\]]{0,120}\]', ' ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove markdown
    text = re.sub(r'\*{1,3}([^*]*)\*{1,3}', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'`[^`]*`', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Truncate at sentence boundary
    if len(text) > 250:
        cut = text[:250]
        last = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
        text = cut[:last + 1] if last > 80 else cut.rstrip() + "."

    return text.strip()

