from dataclasses import dataclass, field
from math import sqrt

from models.base import MemoryRecord

_WILSON_Z = 1.96


def _validate_string_list(values: list[str], *, field_name: str, allow_empty: bool) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a list of strings")
    if not allow_empty and not values:
        raise ValueError(f"{field_name} must contain at least one entry")

    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain only non-blank strings")


@dataclass(kw_only=True)
class ProceduralMemory(MemoryRecord):
    memory_type: str = "procedural"
    steps: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-blank string")
        _validate_string_list(self.steps, field_name="steps", allow_empty=False)
        _validate_string_list(self.preconditions, field_name="preconditions", allow_empty=True)

        if self.success_count < 0 or self.failure_count < 0:
            raise ValueError("success_count and failure_count must be non-negative")

    @property
    def total_outcomes(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_outcomes == 0:
            return 0.0
        return self.success_count / self.total_outcomes

    @property
    def wilson_score(self) -> float:
        n = self.total_outcomes
        if n == 0:
            return 0.0

        p = self.success_count / n
        z_squared = _WILSON_Z**2
        numerator = (
            p
            + z_squared / (2 * n)
            - _WILSON_Z * sqrt((p * (1 - p) / n) + (z_squared / (4 * n**2)))
        )
        denominator = 1 + z_squared / n
        return numerator / denominator

    def record_outcome(self, success: bool) -> None:
        if success:
            self.success_count += 1
            return
        self.failure_count += 1
