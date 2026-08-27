"""Strict output schemas for the intent layer. Anything not
representable here cannot be expressed by either agent — the core safety
property. pydantic v2, extra fields forbidden everywhere."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

THRESHOLD_QS = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)
# Duplicated for schema self-containment; a Task-3 test asserts
# schema.THRESHOLD_QS == knobs.THRESHOLD_QS so the two can never drift.


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicyKnobs(_Strict):
    model: Literal["lstm", "tcn", "transformer", "ae"]
    window: Literal[3, 5, 7]
    threshold_q: Union[Annotated[float, Field(strict=True)], Literal["default"]]
    persistence: Literal[1, 2, 3, 5]

    @field_validator("threshold_q")
    @classmethod
    def _check_on_quantile_ladder(cls, v):
        if v == "default" or v in THRESHOLD_QS:
            return v
        raise ValueError(f"threshold_q must be one of {THRESHOLD_QS} or 'default'")


class SetPolicy(_Strict):
    schema_version: Literal[1]
    action: Literal["set_policy"]
    policy: PolicyKnobs
    intent_understood_as: Optional[str]  # required, no default BY DESIGN: the agent must state what it understood (null allowed, absence is not)
    reason: str = Field(min_length=1, max_length=500)


class Refuse(_Strict):
    schema_version: Literal[1]
    action: Literal["refuse"]
    intent_understood_as: Optional[str] = None
    reason: str = Field(min_length=1, max_length=500)


CompilerOutput = Annotated[Union[SetPolicy, Refuse], Field(discriminator="action")]
_compiler_adapter = TypeAdapter(CompilerOutput)


def parse_compiler_output(text: str):
    """JSON text -> SetPolicy | Refuse (raises pydantic.ValidationError)."""
    return _compiler_adapter.validate_json(text)


def parse_reviewer_verdict(text: str) -> ReviewerVerdict:
    """JSON text -> ReviewerVerdict (raises pydantic.ValidationError)."""
    return ReviewerVerdict.model_validate_json(text)


class ReviewerVerdict(_Strict):
    schema_version: Literal[1]
    verdict: Literal["approve", "revise"]
    checked: list[str] = Field(min_length=1)
    critique: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _revise_needs_critique(self):
        if self.verdict == "revise" and not self.critique.strip():
            raise ValueError("critique required when verdict=revise")
        return self
