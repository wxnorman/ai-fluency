from dataclasses import dataclass
import re
from typing import Literal


ParseMethod = Literal["exact_integer", "equation_rhs", "parse_failure"]

EXACT_INTEGER_PATTERN = re.compile(r"-?[0-9]+")
EQUATION_RHS_PATTERN = re.compile(r".*=\s*(-?[0-9]+)\s*")


@dataclass(frozen=True)
class ParseResult:
    answer: str | None
    method: ParseMethod


def is_format_compliant(completion: str) -> bool:
    """True only when the entire completion is one integer."""
    return EXACT_INTEGER_PATTERN.fullmatch(completion) is not None


def parse_answer(completion: str) -> ParseResult:
    """
    Accepted forms:

    1. Exact integer:
       45
       -44

    2. Equation ending in an integer:
       17 + 28 = 45
       83 - 127 = -44

    Everything else is a parse failure.
    """
    if EXACT_INTEGER_PATTERN.fullmatch(completion) is not None:
        return ParseResult(completion, "exact_integer")

    equation_match = EQUATION_RHS_PATTERN.fullmatch(completion)
    if equation_match is not None:
        return ParseResult(equation_match.group(1), "equation_rhs")

    return ParseResult(None, "parse_failure")
