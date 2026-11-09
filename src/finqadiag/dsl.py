"""Reasoning-program DSL for FinQA-style numerical QA (own implementation).

A *program* is a comma-separated sequence of steps, e.g.

    subtract(920, 95), subtract(469, 77), subtract(#1, #0), divide(#2, #0)

Each step is ``op(arg1, arg2)``.  An argument is one of
  * a number copied from the document        ("920", "5.0", "12%")
  * a reserved constant                       ("const_100", "const_m1")
  * a reference to an earlier step result     ("#0", "#1", ...)
  * a table-row label (table_* ops only)      ("beginning balance"), plus "none"

Ten operators exist: add, subtract, multiply, divide, exp, greater and the
table aggregations table_max / table_min / table_sum / table_average.

This module provides parsing (tolerant of malformed model output), execution
over a table, and a canonical symbolic form used for *program accuracy*.
Nothing here is copied from the official FinQA code; it is written from the
dataset's documented format and validated by re-executing all gold programs
(see scripts/01_validate_dsl.py).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

BINARY_OPS = ("add", "subtract", "multiply", "divide", "exp", "greater")
TABLE_OPS = ("table_max", "table_min", "table_sum", "table_average")
ALL_OPS = BINARY_OPS + TABLE_OPS
COMMUTATIVE_OPS = ("add", "multiply")

EOF = "EOF"
Value = Union[float, str]  # float, or "yes"/"no" for `greater`


class ProgramError(ValueError):
    """The program text/tokens are not a well-formed program."""


class ExecError(ValueError):
    """The program is well-formed but cannot be executed (bad ref, div by 0, ...)."""


@dataclass(frozen=True)
class Step:
    op: str
    args: Tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.op}({', '.join(self.args)})"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_program(text: str) -> List[Step]:
    """Parse program text into steps. Raises ProgramError if malformed.

    Uses a depth-aware scan rather than a regex so that table-row labels that
    themselves contain parentheses (e.g. "revenue ( millions )") do not break
    parsing.
    """
    text = text.strip()
    if not text:
        raise ProgramError("empty program")
    steps: List[Step] = []
    i, n = 0, len(text)
    while i < n:
        m = re.compile(r"\s*([a-z_]+)\(").match(text, i)
        if not m:
            raise ProgramError(f"expected operator at position {i}: {text[i:i+20]!r}")
        op = m.group(1)
        if op not in ALL_OPS:
            raise ProgramError(f"unknown operator {op!r}")
        j = m.end()
        depth, start = 1, j
        while j < n and depth:
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        if depth:
            raise ProgramError("unbalanced parentheses")
        inner = text[start:j - 1]
        args = tuple(_split_top_level(inner))
        want = 2
        if len(args) != want:
            raise ProgramError(f"{op} expects {want} args, got {len(args)}: {inner!r}")
        steps.append(Step(op, args))
        k = re.compile(r"\s*(,|$)").match(text, j)
        if not k:
            raise ProgramError(f"expected ',' after step at position {j}")
        i = k.end()
    return steps


def _split_top_level(s: str) -> List[str]:
    """Split on commas that are not inside parentheses."""
    out, depth, cur = [], 0, []
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur).strip())
    return out


def program_to_tokens(steps: Sequence[Step]) -> List[str]:
    """Flatten steps into a token list: op ( a , b ) , op ( ... ) EOF.

    Arguments are kept as single tokens (a multi-word row label is one token).
    This is the sequence the generator is trained to emit.
    """
    toks: List[str] = []
    for k, st in enumerate(steps):
        if k:
            toks.append(",")
        toks.append(st.op)
        toks.append("(")
        for a_i, a in enumerate(st.args):
            if a_i:
                toks.append(",")
            toks.append(a)
        toks.append(")")
    toks.append(EOF)
    return toks


def parse_tokens(tokens: Sequence[str]) -> List[Step]:
    """Parse a generated token sequence (as produced by program_to_tokens)."""
    toks = list(tokens)
    if toks and toks[-1] == EOF:
        toks = toks[:-1]
    steps: List[Step] = []
    i = 0
    while i < len(toks):
        if toks[i] == ",":  # separator between steps
            i += 1
            continue
        op = toks[i]
        if op not in ALL_OPS:
            raise ProgramError(f"unknown operator token {op!r}")
        try:
            lp, a, comma, b, rp = toks[i + 1:i + 6]
        except ValueError:
            raise ProgramError("truncated program")
        if lp != "(" or comma != "," or rp != ")":
            raise ProgramError("malformed step tokens")
        steps.append(Step(op, (a, b)))
        i += 6
    if not steps:
        raise ProgramError("empty program")
    return steps


# --------------------------------------------------------------------------
# Numbers and tables
# --------------------------------------------------------------------------

_CONST_RE = re.compile(r"^const_(m?\d+)$")


def str_to_num(s: str) -> Optional[float]:
    """Parse a number as it appears in FinQA text/tables; None if not numeric.

    Handles reserved constants (const_100, const_m1), thousands commas,
    leading '$', trailing '%' (value divided by 100) and accounting-style
    negatives in parentheses.
    """
    s = s.strip()
    m = _CONST_RE.match(s)
    if m:
        body = m.group(1)
        return -float(body[1:]) if body.startswith("m") else float(body)
    t = s.replace(",", "").replace("$", "").replace(" ", "")
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1]
    pct = t.endswith("%")
    if pct:
        t = t[:-1]
    try:
        v = float(t)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    v = -v if neg else v
    return v / 100.0 if pct else v


_LEADING_NUM_RE = re.compile(r"^\s*(\$?\s*-?[\d,]*\.?\d+\s*%?)")


def cell_value(cell: str) -> Optional[float]:
    """Numeric value of a table cell, or None.

    The dataset's normalised tables render percentages as e.g. ``22% ( 22 % )``
    (value followed by a parenthesised duplicate), which a whole-cell parse
    rejects; in that case fall back to the leading number token.
    """
    v = str_to_num(cell)
    if v is not None:
        return v
    m = _LEADING_NUM_RE.match(cell)
    return str_to_num(m.group(1)) if m else None


def _row_values(table: Sequence[Sequence[str]], label: str) -> List[float]:
    """Numeric cells of the row named ``label``.

    Some tables stack several blocks with repeated labels (e.g. "fourth
    quarter" under both 2014 and 2013). The dataset's stored answers
    correspond to the LAST matching row, so that is what we use; this is a
    property of the dataset, not a modelling choice, and means such questions
    are inherently ambiguous from the label alone.
    """
    label = label.strip().lower()
    match = None
    for row in table:
        if row and str(row[0]).strip().lower() == label:
            match = row
    if match is None:
        raise ExecError(f"row label {label!r} not found in table")
    vals = [cell_value(str(c)) for c in match[1:]]
    vals = [v for v in vals if v is not None]
    if not vals:
        raise ExecError(f"row {label!r} has no numeric cells")
    return vals


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def execute(steps: Sequence[Step], table: Sequence[Sequence[str]] = ()) -> Value:
    """Run a program; returns the last step's value (float or 'yes'/'no')."""
    results: List[Value] = []

    def resolve(arg: str) -> float:
        if arg.startswith("#"):
            try:
                k = int(arg[1:])
            except ValueError:
                raise ExecError(f"bad reference {arg!r}")
            if not 0 <= k < len(results):
                raise ExecError(f"reference {arg} to a step that does not exist yet")
            v = results[k]
            if isinstance(v, str):
                raise ExecError("arithmetic on a yes/no result")
            return v
        v = str_to_num(arg)
        if v is None:
            raise ExecError(f"not a number: {arg!r}")
        return v

    for st in steps:
        if st.op in TABLE_OPS:
            vals = _row_values(table, st.args[0])
            res: Value = {
                "table_max": max, "table_min": min,
                "table_sum": sum, "table_average": lambda v: sum(v) / len(v),
            }[st.op](vals)
        else:
            a, b = resolve(st.args[0]), resolve(st.args[1])
            try:
                if st.op == "add":
                    res = a + b
                elif st.op == "subtract":
                    res = a - b
                elif st.op == "multiply":
                    res = a * b
                elif st.op == "divide":
                    res = a / b
                elif st.op == "exp":
                    res = a ** b
                else:  # greater
                    res = "yes" if a > b else "no"
            except (ZeroDivisionError, OverflowError, ValueError) as e:
                raise ExecError(str(e))
            if isinstance(res, complex):
                raise ExecError("complex result")
        results.append(res)
    if not results:
        raise ExecError("no steps")
    return results[-1]


def try_execute(steps_or_text, table=()) -> Optional[Value]:
    """Like execute() but returns None for any malformed/unexecutable program."""
    try:
        steps = parse_program(steps_or_text) if isinstance(steps_or_text, str) else steps_or_text
        return execute(steps, table)
    except (ProgramError, ExecError):
        return None


# --------------------------------------------------------------------------
# Symbolic canonical form (for program accuracy)
# --------------------------------------------------------------------------

def _norm_leaf(arg: str) -> str:
    v = str_to_num(arg)
    if v is not None:
        return repr(round(v, 8))
    return arg.strip().lower()


def _mask_leaf(arg: str) -> str:
    """Leaf normaliser for skeletons: keep structure-defining tokens, mask data.

    Reserved constants (const_100, const_m1) and 'none' are part of the
    program's structure; numbers copied from the document become NUM and
    table-row labels become ROW.
    """
    a = arg.strip()
    if a.startswith("const_") or a == "none":
        return a
    return "NUM" if str_to_num(a) is not None else "ROW"


def skeleton(steps: Sequence[Step]) -> str:
    """Canonical form with document operands masked (operators + constants only)."""
    return canonical(steps, leaf=_mask_leaf)


def canonical(steps: Sequence[Step], leaf=_norm_leaf) -> str:
    """Nested symbolic form of the *final* step.

    Step references are expanded into sub-expressions and arguments of
    commutative operators (add, multiply) are sorted, so two programs that
    differ only in operand order or in the order of independent steps map to
    the same string.  ``leaf`` normalises non-reference arguments.
    """
    memo: List[str] = []
    for k, st in enumerate(steps):
        kids = []
        for a in st.args:
            if a.startswith("#"):
                try:
                    j = int(a[1:])
                except ValueError:
                    raise ProgramError(f"bad reference {a!r}")
                if not 0 <= j < k:
                    raise ProgramError(f"forward/invalid reference {a!r}")
                kids.append(memo[j])
            else:
                kids.append(leaf(a))
        if st.op in COMMUTATIVE_OPS:
            kids.sort()
        memo.append(f"{st.op}({','.join(kids)})")
    return memo[-1]
