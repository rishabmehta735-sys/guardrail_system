from dataclasses import dataclass, field
from enum import Enum
 
 
class Verdict(Enum):
    ALLOW    = "ALLOW"
    BLOCK    = "BLOCK"
    FLAG     = "FLAG"
    ESCALATE = "ESCALATE"
    MODIFY   = "MODIFY"
 
 
@dataclass
class Decision:
    verdict:    Verdict
    stage:      str
    reason:     str
    confidence: float          # 0.0 – 1.0
    metadata:   dict = field(default_factory=dict)