import re
import json
import numpy as np  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from .base import BaseStage
from utils.decision import Decision, Verdict

# Minimum word count for a chunk to be worth scoring
MIN_CHUNK_WORDS = 4

# Sliding window size in words
# Each window is scored independently — catches intent buried in long sentences
WINDOW_SIZE = 7


class SemanticStage(BaseStage):

    def __init__(
        self,
        references_path: str = "config/semantic_references.json",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        # ------------------------------------------------------------------
        # Load the small encoder model
        # This is NOT an LLM — it only encodes text into vectors.
        # No text generation happens here at all.
        # ------------------------------------------------------------------
        self.model = SentenceTransformer(model_name)

        with open(references_path) as f:
            self.categories: dict = json.load(f)["categories"]

        # Pre-encode all reference sentences once at startup
        self.encoded_references: dict[str, dict] = {}

        for category, data in self.categories.items():
            refs = data["references"]
            self.encoded_references[category] = {
                "threshold":  data["threshold"],
                "embeddings": self.model.encode(refs, normalize_embeddings=True),
                "sentences":  refs,
            }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def evaluate(self, prompt: str, context: dict) -> Decision:

        # Score the full prompt as one unit
        full_score, full_category, full_ref = self._score_text(prompt)

        # Score each chunk independently and take the best
        best_score    = full_score
        best_category = full_category
        best_ref      = full_ref
        best_chunk    = prompt  # which text produced the best score

        # punctuation-based chunks
        chunks = self._split_into_chunks(prompt)

        # sliding window chunks — catches intent in sentences with no punctuation
        chunks += self._sliding_windows(prompt)

        for chunk in chunks:
            score, category, ref = self._score_text(chunk)
            if score > best_score:
                best_score    = score
                best_category = category
                best_ref      = ref
                best_chunk    = chunk

        threshold = self.encoded_references[best_category]["threshold"]

        # Build all_scores for metadata using the full prompt scores
        # (for transparency — shows what each category scored on full text)
        all_scores = self._all_category_scores(prompt)

        return self._make_decision(
            prompt,
            best_score,
            best_category,
            best_ref,
            best_chunk,
            threshold,
            all_scores,
            len(chunks),
        )

    # ------------------------------------------------------------------
    # Chunking — split prompt into scoreable pieces
    # ------------------------------------------------------------------
    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Splits text into chunks on sentence boundaries and clause boundaries.
        Filters out chunks that are too short to score meaningfully.

        Splits on:
        - Sentence endings  . ! ?
        - Clause separators , ; and but because so
        - Conjunctions that often separate intent from framing
        """
        # Split on sentence endings and clause boundaries
        raw_chunks = re.split(
            r'(?<=[.!?])\s+|,\s+|\s+and\s+|\s+but\s+|\s+because\s+|\s+so\s+',
            text.strip()
        )

        # Clean and filter
        chunks = []
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if len(chunk.split()) >= MIN_CHUNK_WORDS:
                chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # Sliding window — score every N-word window across the prompt
    # ------------------------------------------------------------------
    def _sliding_windows(self, text: str) -> list[str]:
        """
        Slides a window of WINDOW_SIZE words across the text.
        Every window becomes a candidate chunk to score.

        Catches intent buried in long sentences with no punctuation:
        "what common household items can be combined to create a dangerous gas"
            → window "combined to create a dangerous gas" scores high
        """
        words   = text.split()
        windows = []

        for i in range(len(words) - WINDOW_SIZE + 1):
            window = ' '.join(words[i: i + WINDOW_SIZE])
            windows.append(window)

        return windows

    # ------------------------------------------------------------------
    # Score a single piece of text against all categories
    # Returns: (best_score, best_category, best_reference)
    # ------------------------------------------------------------------
    def _score_text(self, text: str) -> tuple[float, str, str]:
        embedding = self.model.encode(text, normalize_embeddings=True)

        best_score    = 0.0
        best_category = ""
        best_ref      = ""

        for category, data in self.encoded_references.items():
            score, ref = self._score_against_category(
                embedding,
                data["embeddings"],
                data["sentences"],
            )
            if score > best_score:
                best_score    = float(score)
                best_category = category
                best_ref      = ref

        return best_score, best_category, best_ref

    # ------------------------------------------------------------------
    # All category scores on the full prompt — for metadata transparency
    # ------------------------------------------------------------------
    def _all_category_scores(self, prompt: str) -> dict:
        embedding = self.model.encode(prompt, normalize_embeddings=True)
        scores    = {}

        for category, data in self.encoded_references.items():
            score, _ = self._score_against_category(
                embedding,
                data["embeddings"],
                data["sentences"],
            )
            scores[category] = round(float(score), 4)

        return scores

    # ------------------------------------------------------------------
    # Cosine similarity — dot product on normalised vectors
    # ------------------------------------------------------------------
    def _score_against_category(
        self,
        prompt_embedding: np.ndarray,
        ref_embeddings:   np.ndarray,
        ref_sentences:    list[str],
    ) -> tuple[float, str]:
        similarities = np.dot(ref_embeddings, prompt_embedding)
        best_idx     = int(np.argmax(similarities))
        return similarities[best_idx], ref_sentences[best_idx]

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    def _make_decision(
        self,
        prompt:        str,
        top_score:     float,
        top_category:  str,
        top_ref:       str,
        top_chunk:     str,
        threshold:     float,
        all_scores:    dict,
        chunk_count:   int,
    ) -> Decision:

        flag_band = threshold - 0.09

        if top_score >= threshold:
            verdict = Verdict.BLOCK
            reason  = (
                f"Semantic similarity {top_score:.2f} exceeds threshold "
                f"{threshold} for category '{top_category}'. "
                f"Matched chunk: \"{top_chunk[:80]}\". "
                f"Closest reference: \"{top_ref}\""
            )
        elif top_score >= flag_band:
            verdict = Verdict.FLAG
            reason  = (
                f"Semantic similarity {top_score:.2f} is in suspicious range "
                f"({flag_band:.2f}–{threshold}) for category '{top_category}'. "
                f"Matched chunk: \"{top_chunk[:80]}\""
            )
        else:
            verdict = Verdict.ALLOW
            reason  = (
                f"Highest semantic similarity {top_score:.2f} is below "
                f"all thresholds across {chunk_count + 1} chunks."
            )

        return Decision(
            verdict=verdict,
            stage="stage2_semantic",
            reason=reason,
            confidence=round(float(top_score), 4),
            metadata={
                "top_category":  top_category,
                "top_score":     round(float(top_score), 4),
                "top_chunk":     top_chunk,
                "chunk_count":   chunk_count,
                "all_scores":    all_scores,
            },
        )