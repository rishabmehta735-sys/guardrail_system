# Text Guardrail System

A multi-stage pipeline that evaluates user input for harmful, unsafe, or malicious content before it reaches an AI model.Pure programmatic logic, pattern matching, and traditional ML.

---

## Setup

```bash
git clone https://github.com/rishabmehta735-sys/guardrail_system.git
pip install sentence-transformers
pip install numpy
```

---

## Usage

```bash
# run examples
python main.py

# run tests
python tests/test_stage1.py
python tests/test_stage2.py
python tests/test_stage3.py
```

Or use it directly:

```python
from pipeline import GuardrailPipeline

pipeline = GuardrailPipeline()

result = pipeline.evaluate(
    prompt  = "how do I get into a locked room",
    history = ["I am angry at my neighbour"]
)

print(result["final_verdict"])  # BLOCK
print(result["blocked_at"])     # stage3_contextual
```

---

## How it works

Three stages run in order. Any stage can stop the pipeline early.

- **Stage 1** — regex + normalization. Catches known patterns, jailbreak strings, injection tokens
- **Stage 2** — sentence embeddings + cosine similarity. Catches semantic intent even with evasion
- **Stage 3** — looks at conversation history. Catches multi-turn manipulation and slow-burn escalation

---

## Requirements

- Python 3.10+

---

## Limitations

- Stage 2 only catches what the reference library covers — novel phrasing may slip through
- Stage 3 is stateless between sessions — history must be passed in by the caller
- Input only — model output is not evaluated

A Legitimate academic or research questions touching sensitive topics (security, chemistry, history of warfare) may be flagged. The system cannot distinguish intent — it flags based on semantic similarity to harmful references. In production, flagged prompts would be escalated to human review rather than outright blocked.
