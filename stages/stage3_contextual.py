from .base import BaseStage
from .stage1_lexical import LexicalStage
from .stage2_semantic import SemanticStage
from utils.decision import Decision, Verdict
 
 
# ------------------------------------------------------------------
# Weights
# ------------------------------------------------------------------
STAGE1_HIT_WEIGHT     = 0.70
STAGE2_SCORE_WEIGHT   = 1.00
REPEAT_CATEGORY_BONUS = 0.30
 
# ------------------------------------------------------------------
# Per-category noise floors
# ------------------------------------------------------------------
CATEGORY_NOISE_FLOORS = {
    "jailbreak_and_bypass":       0.55,
    "hacking_and_cyberattacks":   0.45,
    "sensitive_data_extraction":  0.50,
    "weapons_and_explosives":     0.38,
    "self_harm_and_violence":     0.38,
    "illegal_acquisition":        0.42,
    "theft_and_physical_intrusion": 0.38,
    "grooming_and_child_safety":     0.45,
    "privacy_violation":             0.40,
}
DEFAULT_NOISE_FLOOR = 0.30
 
# ------------------------------------------------------------------
# History window
# ------------------------------------------------------------------
HISTORY_WINDOW = 6
 
# ------------------------------------------------------------------
# Base thresholds — scaled down for short conversations
# ------------------------------------------------------------------
BASE_BLOCK_THRESHOLD = 1.30
BASE_FLAG_THRESHOLD  = 0.80
MIN_SCALE            = 0.50   # shortest conversation gets 40% of threshold
 
 
def _scaled_threshold(base: float, total_turns: int) -> float:
    """
    Scale threshold down for short conversations.
    1 turn  → base * 0.40
    3 turns → base * 0.60
    6 turns → base * 1.00 (full threshold)
    """
    max_turns = HISTORY_WINDOW
    scale     = MIN_SCALE + (1.0 - MIN_SCALE) * min(total_turns / max_turns, 1.0)
    return round(base * scale, 4)
 
 
class ContextualStage(BaseStage):
 
    def __init__(self):
        self.stage1 = LexicalStage()
        self.stage2 = SemanticStage()
 
    def evaluate(self, prompt: str, context: dict) -> Decision:
        history        = context.get("history", [])
        recent_history = history[-HISTORY_WINDOW:]
        empty_context  = {"history": [], "scores": []}
 
        stage1_hits     = 0
        stage2_total    = 0.0
        category_counts = {}
        turn_details    = []
 
        # ------------------------------------------------------------------
        # Score history messages
        # ------------------------------------------------------------------
        for message in recent_history:
            s1_decision = self.stage1.evaluate(message, empty_context)
            s1_hit      = s1_decision.verdict == Verdict.BLOCK
 
            if s1_hit:
                stage1_hits += 1
 
            s2_decision = self.stage2.evaluate(message, empty_context)
            s2_score    = s2_decision.confidence
            s2_category = s2_decision.metadata.get("top_category", "unknown")
            noise_floor = CATEGORY_NOISE_FLOORS.get(s2_category, DEFAULT_NOISE_FLOOR)
            counted     = s2_score >= noise_floor
 
            if counted:
                stage2_total += s2_score
                category_counts[s2_category] = (
                    category_counts.get(s2_category, 0) + 1
                )
 
            turn_details.append({
                "message":     message[:60],
                "s1_hit":      s1_hit,
                "s2_score":    round(s2_score, 4),
                "s2_category": s2_category,
                "counted":     counted,
            })
 
        # ------------------------------------------------------------------
        # Score current prompt — same weight as history, no multiplier
        # ------------------------------------------------------------------
        s1_current       = self.stage1.evaluate(prompt, empty_context)
        s2_current       = self.stage2.evaluate(prompt, empty_context)
        current_s1_hit   = s1_current.verdict == Verdict.BLOCK
        current_s2_score = s2_current.confidence
        current_category = s2_current.metadata.get("top_category", "unknown")
        current_floor    = CATEGORY_NOISE_FLOORS.get(current_category, DEFAULT_NOISE_FLOOR)
        current_counted  = current_s2_score >= current_floor
 
        if current_s1_hit:
            stage1_hits += 1
 
        if current_counted:
            stage2_total += current_s2_score
            category_counts[current_category] = (
                category_counts.get(current_category, 0) + 1
            )
 
        turn_details.append({
            "message":     f"[CURRENT] {prompt[:60]}",
            "s1_hit":      current_s1_hit,
            "s2_score":    round(current_s2_score, 4),
            "s2_category": current_category,
            "counted":     current_counted,
        })
 
        # ------------------------------------------------------------------
        # Repeat category bonus
        # ------------------------------------------------------------------
        repeat_bonus = sum(
            (count - 1) * REPEAT_CATEGORY_BONUS
            for count in category_counts.values()
            if count > 1
        )
 
        # ------------------------------------------------------------------
        # Risk score
        # ------------------------------------------------------------------
        risk_score = round(
            (stage1_hits  * STAGE1_HIT_WEIGHT)
          + (stage2_total * STAGE2_SCORE_WEIGHT)
          + repeat_bonus,
            4
        )
 
        # ------------------------------------------------------------------
        # Scale thresholds based on total turns (history + current)
        # ------------------------------------------------------------------
        total_turns     = len(recent_history) + 1
        block_threshold = BASE_BLOCK_THRESHOLD
        flag_threshold  = _scaled_threshold(BASE_FLAG_THRESHOLD,  total_turns)
 
        dominant_category = (
            max(category_counts, key=category_counts.get)
            if category_counts else "none"
        )
 
        return self._make_decision(
            risk_score,
            block_threshold,
            flag_threshold,
            dominant_category,
            stage1_hits,
            stage2_total,
            repeat_bonus,
            turn_details,
            len(recent_history),
        )
 
    def _make_decision(
        self,
        risk_score:        float,
        block_threshold:   float,
        flag_threshold:    float,
        dominant_category: str,
        stage1_hits:       int,
        stage2_total:      float,
        repeat_bonus:      float,
        turn_details:      list,
        history_length:    int,
    ) -> Decision:
 
        if risk_score >= block_threshold:
            verdict = Verdict.BLOCK
            reason  = (
                f"Cumulative risk score {risk_score:.2f} exceeds scaled block "
                f"threshold {block_threshold:.2f}. "
                f"Dominant category: '{dominant_category}'. "
                f"Stage1 hits: {stage1_hits}. "
                f"Semantic total: {stage2_total:.2f}."
            )
        elif risk_score >= flag_threshold:
            verdict = Verdict.FLAG
            reason  = (
                f"Cumulative risk score {risk_score:.2f} is in suspicious "
                f"range ({flag_threshold:.2f}–{block_threshold:.2f}). "
                f"Dominant category: '{dominant_category}'."
            )
        else:
            verdict = Verdict.ALLOW
            reason  = (
                f"Cumulative risk score {risk_score:.2f} is below scaled flag "
                f"threshold {flag_threshold:.2f}. Conversation appears safe."
            )
 
        return Decision(
            verdict=verdict,
            stage="stage3_contextual",
            reason=reason,
            confidence=round(min(risk_score / block_threshold, 1.0), 4),
            metadata={
                "risk_score":        risk_score,
                "block_threshold":   block_threshold,
                "flag_threshold":    flag_threshold,
                "dominant_category": dominant_category,
                "stage1_hits":       stage1_hits,
                "stage2_total":      round(stage2_total, 4),
                "repeat_bonus":      round(repeat_bonus, 4),
                "history_length":    history_length,
                "turn_details":      turn_details,
            },
        )
 