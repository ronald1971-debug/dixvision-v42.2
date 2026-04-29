"""Tests for the authority matrix loader and the canonical YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from system_engine.authority import load_authority_matrix
from system_engine.authority.matrix import AuthorityMatrix

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "registry" / "authority_matrix.yaml"


# ---------------------------------------------------------------------------
# canonical file
# ---------------------------------------------------------------------------


def test_canonical_matrix_loads():
    m = load_authority_matrix(CANONICAL)
    assert isinstance(m, AuthorityMatrix)
    # Triad lock invariant: governance / intelligence / execution all present.
    ids = m.actor_ids
    assert {"governance", "intelligence", "execution", "system", "operator"} <= ids


def test_canonical_precedence_governance_above_engines():
    m = load_authority_matrix(CANONICAL)
    assert m.precedence_index("governance") < m.precedence_index("intelligence")
    assert m.precedence_index("governance") < m.precedence_index("execution")
    assert m.precedence_index("ledger") < m.precedence_index("governance")


def test_canonical_overrides_route_through_governance():
    m = load_authority_matrix(CANONICAL)
    for ovr in m.overrides:
        assert ovr.via == "governance", (
            f"override {ovr.id} must route through governance"
        )


def test_canonical_resolve_picks_higher_precedence():
    m = load_authority_matrix(CANONICAL)
    assert m.resolve("intelligence", "governance") == "governance"
    assert m.resolve("execution", "system") == "system"
    assert m.resolve("learning", "evolution") == "learning"


def test_canonical_conflict_ids_unique_and_namespaced():
    m = load_authority_matrix(CANONICAL)
    ids = [c.id for c in m.conflicts]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("CONF-") for i in ids)


# ---------------------------------------------------------------------------
# loader edge cases (rejection paths)
# ---------------------------------------------------------------------------


def _write(tmp_path, body):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def _good_body() -> dict:
    return {
        "version": "v0.0.1",
        "actors": [
            {"id": "governance", "role": "x", "module": "governance_engine"},
            {"id": "intelligence", "role": "x", "module": "intelligence_engine"},
        ],
        "precedence": ["governance", "intelligence"],
        "conflicts": [
            {
                "id": "CONF-A",
                "domain": "x",
                "description": "x",
                "winner": "governance",
            }
        ],
        "overrides": [
            {
                "id": "OVR-A",
                "name": "n",
                "grants": "intelligence",
                "overrides": ["governance"],
                "via": "governance",
            }
        ],
    }


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_authority_matrix(tmp_path / "nope.yaml")


def test_load_rejects_unknown_winner(tmp_path):
    body = _good_body()
    body["conflicts"][0]["winner"] = "alien"
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="winner 'alien'"):
        load_authority_matrix(p)


def test_load_accepts_deferred_winner(tmp_path):
    body = _good_body()
    body["conflicts"][0]["winner"] = "deferred"
    p = _write(tmp_path, body)
    m = load_authority_matrix(p)
    assert m.conflicts[0].winner == "deferred"


def test_load_rejects_precedence_missing_actor(tmp_path):
    body = _good_body()
    body["precedence"] = ["governance"]  # intelligence missing
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="not covered by precedence"):
        load_authority_matrix(p)


def test_load_rejects_precedence_unknown_actor(tmp_path):
    body = _good_body()
    body["precedence"].append("ghost")
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="unknown actors"):
        load_authority_matrix(p)


def test_load_rejects_duplicate_actor_ids(tmp_path):
    body = _good_body()
    body["actors"].append(
        {"id": "governance", "role": "x", "module": "governance_engine"}
    )
    body["precedence"] = ["governance", "intelligence"]
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="duplicate actor ids"):
        load_authority_matrix(p)


def test_load_rejects_self_override(tmp_path):
    body = _good_body()
    body["overrides"][0]["overrides"] = ["intelligence"]  # self-override
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="cannot override itself"):
        load_authority_matrix(p)


def test_load_rejects_via_other_than_governance(tmp_path):
    body = _good_body()
    body["overrides"][0]["via"] = "intelligence"
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="via must be 'governance'"):
        load_authority_matrix(p)


def test_load_rejects_unknown_override_target(tmp_path):
    body = _good_body()
    body["overrides"][0]["overrides"] = ["alien"]
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="overrides references unknown actors"):
        load_authority_matrix(p)


def test_load_rejects_duplicate_conflict_ids(tmp_path):
    body = _good_body()
    body["conflicts"].append(dict(body["conflicts"][0]))
    p = _write(tmp_path, body)
    with pytest.raises(ValueError, match="duplicate conflict ids"):
        load_authority_matrix(p)


def test_load_rejects_missing_top_level_keys(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"version": "v0"}))
    with pytest.raises(ValueError, match="missing required top-level keys"):
        load_authority_matrix(p)


def test_resolve_uses_precedence_order(tmp_path):
    body = _good_body()
    p = _write(tmp_path, body)
    m = load_authority_matrix(p)
    assert m.resolve("governance", "intelligence") == "governance"
    assert m.resolve("intelligence", "governance") == "governance"


def test_actor_lookup_unknown_raises():
    m = load_authority_matrix(CANONICAL)
    with pytest.raises(KeyError):
        m.actor("ghost")
