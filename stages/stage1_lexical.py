import re
import json
import unicodedata
from .base import BaseStage
from utils.decision import Decision, Verdict
 

LEET_MAP = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a',
    '5': 's', '6': 'g', '7': 't', '8': 'b',
    '@': 'a', '$': 's', '!': 'i', '+': 't',
}
class LexicalStage(BaseStage):
 
    def __init__(self, patterns_path: str = "config/patterns.json"):
        with open(patterns_path) as f:
            data = json.load(f)
 
        # Each category is a list of regex patterns
        self.categories: dict[str, list[str]] = data["categories"]
 
    # ------------------------------------------------------------------
    # Public entry point — called by the pipeline
    # ------------------------------------------------------------------
    def evaluate(self, prompt: str,context: dict) -> Decision:
        normalized = self._normalize(prompt)
 
        for category, patterns in self.categories.items():
            for pattern in patterns:
                if re.search(pattern, normalized, re.IGNORECASE):
                    return Decision(
                        verdict=Verdict.BLOCK,
                        stage="stage1_lexical",
                        reason=f"Matched category '{category}' — pattern: {pattern}",
                        confidence=1.0,
                        metadata={
                            "category": category,
                            "matched_pattern": pattern,
                            "normalized_prompt": normalized,
                        },
                    )
 
        return Decision(
            verdict=Verdict.ALLOW,
            stage="stage1_lexical",
            reason="No lexical patterns matched",
            confidence=1.0,
            metadata={"normalized_prompt": normalized},
        )
 
    # ------------------------------------------------------------------
    # Normalization pipeline
    # Order matters: unicode → whitespace → leetspeak → lowercase
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        text = self._strip_unicode_tricks(text)
        text = self._remove_invisible_chars(text)
        text = self._normalize_whitespace(text)
        text = self._decode_leetspeak(text)  # ← add this back
        text = text.lower().strip()
        return text
 
    def _strip_unicode_tricks(self, text: str) -> str:
        """
        Converts fullwidth unicode characters to ASCII equivalents.
        e.g. ｉｇｎｏｒｅ → ignore
        unicodedata.normalize NFKC handles this decomposition.
        """
        return unicodedata.normalize("NFKC", text)
 
    def _remove_invisible_chars(self, text: str) -> str:
        """
        Replaces zero-width and invisible characters with a space
        so that word boundaries are preserved after removal.
        e.g. "ignore​previous" → "ignore previous"
        Replacing with space (not empty string) prevents merging:
        "ignore​previous" → "ignore previous" not "ignoreprevious"
        """
        invisible = [
            '\u200b',  # zero-width space
            '\u200c',  # zero-width non-joiner
            '\u200d',  # zero-width joiner
            '\u00ad',  # soft hyphen
            '\ufeff',  # zero-width no-break space (BOM)
            '\u2060',  # word joiner
        ]
        for char in invisible:
            text = text.replace(char, ' ')
        return text
 
    def _normalize_whitespace(self, text: str) -> str:
        """
        Two passes:
        1. Collapse single-character tokens separated by spaces
           e.g. "i g n o r e" → "ignore"
           Detects when every token between spaces is 1 char long
        2. Collapse multiple spaces into one
        """
        # Detect spaced-out characters: "i g n o r e p r e v i o u s"
        # Pattern: sequences of (single-char + space) repeated 3+ times
        text = re.sub(r'(?<!\S)((\S ){3,}\S)(?!\S)', self._collapse_spaced, text)
        return re.sub(r'\s+', ' ', text)
 
    @staticmethod
    def _collapse_spaced(match: re.Match) -> str:
        """Removes spaces from a spaced-character sequence."""
        return match.group(0).replace(' ', '')
    
    def _decode_leetspeak(self, text: str) -> str:
        return ''.join(LEET_MAP.get(c, c) for c in text)
 
