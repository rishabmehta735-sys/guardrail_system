
from abc import ABC, abstractmethod
from utils.decision import Decision
 
 
class BaseStage(ABC):
 
    @abstractmethod
    def evaluate(self, prompt: str, context: dict) -> Decision:
        """
        prompt  — the current user input (raw)
        context — dict containing:
                    history : list[str]  → prior conversation turns
                    scores  : list[float] → confidence scores from earlier stages
        returns — a Decision object
        """
        pass