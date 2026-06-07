from stages.stage1_lexical    import LexicalStage
from stages.stage2_semantic   import SemanticStage
from stages.stage3_contextual import ContextualStage
from utils.decision import Decision, Verdict


class GuardrailPipeline:

    def __init__(self):
        # Initialise all stages once
        # Stage 2 loads the sentence-transformer model here — once only
        print("[Pipeline] Initialising stages...")
        self.stage1 = LexicalStage()
        self.stage2 = SemanticStage()
        self.stage3 = ContextualStage()
        print("[Pipeline] Ready.\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def evaluate(self, prompt: str, history: list[str] = []) -> dict:
        """
        Runs the prompt through all stages in order.
        Stops early if any stage returns BLOCK.

        Args:
            prompt  — the current user message
            history — list of prior messages in the conversation (oldest first)

        Returns:
            A structured dict with final verdict and all stage results
        """
        context = {
            "history": history,
            "scores":  [],
        }

        results: list[Decision] = []

        # ------------------------------------------------------------------
        # Run each stage in order
        # ------------------------------------------------------------------
        for stage in [self.stage1, self.stage2, self.stage3]:
            decision = stage.evaluate(prompt, context)
            results.append(decision)

            # Pass this stage's confidence forward for later stages to use
            context["scores"].append(decision.confidence)

            # Short circuit — stop pipeline on hard block
            if decision.verdict == Verdict.BLOCK:
                return self._build_output(prompt, history, results, decision)

        # All stages passed — final decision is whatever Stage 3 returned
        return self._build_output(prompt, history, results, results[-1])

    # ------------------------------------------------------------------
    # Build structured output
    # ------------------------------------------------------------------
    def _build_output(
        self,
        prompt:   str,
        history:  list[str],
        results:  list[Decision],
        final:    Decision,
    ) -> dict:
        return {
            "prompt":        prompt,
            "history_turns": len(history),
            "final_verdict": final.verdict.value,
            "final_reason":  final.reason,
            "blocked_at":    final.stage if final.verdict == Verdict.BLOCK else None,
            "stage_results": [
                {
                    "stage":      d.stage,
                    "verdict":    d.verdict.value,
                    "confidence": d.confidence,
                    "reason":     d.reason,
                }
                for d in results
            ],
        }