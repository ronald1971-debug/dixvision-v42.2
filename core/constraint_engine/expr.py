"""Tiny pure expression DSL for constraint-rule predicates.

Grammar (whitespace-insensitive)::

    expr     := or_expr
    or_expr  := and_expr ( "or" and_expr )*
    and_expr := unary    ( "and" unary )*
    unary    := "not" unary | atom
    atom     := "(" expr ")" | comparison
    comparison := operand op operand
    op       := "==" | "!=" | "<" | "<=" | ">" | ">="
    operand  := number | ident
    ident    := [a-zA-Z_][a-zA-Z0-9_]*

The DSL is intentionally minimal — it can compare two fact fields or a
fact field against a numeric literal. It cannot call functions, read
external state, mutate anything, or produce side effects. That is the
property that lets the constraint compiler statically validate rules
without invoking arbitrary code.

Pure / deterministic. INV-15 — no clock, no PRNG, no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Number:
    value: float


@dataclass(frozen=True, slots=True)
class _Ident:
    name: str


@dataclass(frozen=True, slots=True)
class _Cmp:
    op: str
    left: _Number | _Ident
    right: _Number | _Ident


@dataclass(frozen=True, slots=True)
class _Not:
    inner: Expr


@dataclass(frozen=True, slots=True)
class _And:
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class _Or:
    left: Expr
    right: Expr


Expr = _Cmp | _Not | _And | _Or


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    \s* (?:
        (?P<number>-?\d+(?:\.\d+)?) |
        (?P<ident>[A-Za-z_][A-Za-z0-9_]*) |
        (?P<op>==|!=|<=|>=|<|>) |
        (?P<lparen>\() |
        (?P<rparen>\))
    )
    """,
    re.VERBOSE,
)

_KEYWORDS = {"and", "or", "not"}


def _tokenize(src: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(src):
        if src[i].isspace():
            i += 1
            continue
        m = _TOKEN_RE.match(src, i)
        if m is None or m.end() == i:
            raise ValueError(f"unexpected character at offset {i}: {src[i]!r}")
        if m.group("number") is not None:
            tokens.append(("number", m.group("number")))
        elif m.group("ident") is not None:
            tok = m.group("ident")
            if tok in _KEYWORDS:
                tokens.append((tok, tok))
            else:
                tokens.append(("ident", tok))
        elif m.group("op") is not None:
            tokens.append(("op", m.group("op")))
        elif m.group("lparen") is not None:
            tokens.append(("lparen", "("))
        elif m.group("rparen") is not None:
            tokens.append(("rparen", ")"))
        i = m.end()
    tokens.append(("eof", ""))
    return tokens


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> tuple[str, str]:
        return self._tokens[self._pos]

    def _advance(self) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str) -> tuple[str, str]:
        tok = self._advance()
        if tok[0] != kind:
            raise ValueError(f"expected {kind}, got {tok[0]!r} ({tok[1]!r})")
        return tok

    def parse(self) -> Expr:
        node = self._parse_or()
        if self._peek()[0] != "eof":
            raise ValueError(f"trailing tokens after expression: {self._peek()!r}")
        return node

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._peek()[0] == "or":
            self._advance()
            right = self._parse_and()
            left = _Or(left, right)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_unary()
        while self._peek()[0] == "and":
            self._advance()
            right = self._parse_unary()
            left = _And(left, right)
        return left

    def _parse_unary(self) -> Expr:
        if self._peek()[0] == "not":
            self._advance()
            return _Not(self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> Expr:
        tok = self._peek()
        if tok[0] == "lparen":
            self._advance()
            inner = self._parse_or()
            self._expect("rparen")
            return inner
        return self._parse_cmp()

    def _parse_cmp(self) -> _Cmp:
        left = self._parse_operand()
        op_tok = self._advance()
        if op_tok[0] != "op":
            raise ValueError(
                f"expected comparison operator, got {op_tok[0]!r} ({op_tok[1]!r})"
            )
        right = self._parse_operand()
        return _Cmp(op_tok[1], left, right)

    def _parse_operand(self) -> _Number | _Ident:
        tok = self._advance()
        if tok[0] == "number":
            return _Number(float(tok[1]))
        if tok[0] == "ident":
            return _Ident(tok[1])
        raise ValueError(
            f"expected operand (number/ident), got {tok[0]!r} ({tok[1]!r})"
        )


def parse(src: str) -> Expr:
    """Parse an expression string into an immutable AST."""

    return _Parser(_tokenize(src)).parse()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


_NUMERIC_OPS = {"<", "<=", ">", ">="}


def _resolve(
    operand: _Number | _Ident, facts: Mapping[str, Any], *, numeric_only: bool
) -> Any:
    if isinstance(operand, _Number):
        return operand.value
    if operand.name not in facts:
        raise KeyError(f"missing fact {operand.name!r}")
    val = facts[operand.name]
    if isinstance(val, bool):  # bool is a subclass of int — guard explicitly
        return float(int(val))
    if isinstance(val, (int, float)):
        return float(val)
    if numeric_only:
        raise TypeError(
            f"fact {operand.name!r} must be numeric, got {type(val).__name__}"
        )
    if isinstance(val, str):
        return val
    raise TypeError(
        f"fact {operand.name!r} must be numeric or string, got {type(val).__name__}"
    )


_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def evaluate(expr: Expr, facts: Mapping[str, Any]) -> bool:
    """Evaluate a parsed expression against a typed fact mapping."""

    if isinstance(expr, _Cmp):
        numeric_only = expr.op in _NUMERIC_OPS
        left = _resolve(expr.left, facts, numeric_only=numeric_only)
        right = _resolve(expr.right, facts, numeric_only=numeric_only)
        return _OPS[expr.op](left, right)
    if isinstance(expr, _Not):
        return not evaluate(expr.inner, facts)
    if isinstance(expr, _And):
        return evaluate(expr.left, facts) and evaluate(expr.right, facts)
    if isinstance(expr, _Or):
        return evaluate(expr.left, facts) or evaluate(expr.right, facts)
    raise TypeError(f"unknown expression node: {type(expr).__name__}")


def free_idents(expr: Expr) -> frozenset[str]:
    """Return the set of fact names referenced by ``expr``."""

    if isinstance(expr, _Cmp):
        return frozenset(
            o.name for o in (expr.left, expr.right) if isinstance(o, _Ident)
        )
    if isinstance(expr, _Not):
        return free_idents(expr.inner)
    if isinstance(expr, (_And, _Or)):
        return free_idents(expr.left) | free_idents(expr.right)
    raise TypeError(f"unknown expression node: {type(expr).__name__}")


__all__ = ["Expr", "evaluate", "free_idents", "parse"]
