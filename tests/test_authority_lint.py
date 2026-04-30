"""Phase E0 — authority_lint rule unit tests.

Each lint rule (T1, C2, C3, W1, L1, L2, L3, B1) is exercised against
synthetic source trees built in temp directories. The tests guarantee:

* the real repo passes ``authority_lint`` with zero violations;
* every rule fires on a synthetic violation;
* every rule does NOT fire on its allow-listed shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.authority_lint import lint_repo

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    # Minimal package skeleton so AST parses + module names resolve.
    for pkg in (
        "core",
        "core/contracts",
        "state",
        "state/ledger",
        "intelligence_engine",
        "execution_engine",
        "system_engine",
        "governance_engine",
        "learning_engine",
        "evolution_engine",
        "mind",
        "mind/neuromorphic",
        "mind/web_autolearn",
        "execution_engine/adapters",
        "execution_engine/adapters/memecoin",
        "wallet",
    ):
        _write(tmp_path, f"{pkg}/__init__.py", "")
    return tmp_path


def _rule_codes(violations) -> list[str]:
    return [v.rule for v in violations]


def test_real_repo_passes_zero_violations():
    violations = lint_repo(REPO_ROOT)
    assert violations == [], "\n".join(
        v.format(REPO_ROOT) for v in violations
    )


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------


def test_t1_fires_on_hot_path_to_governance(fake_repo: Path):
    _write(fake_repo, "mind/fast_execute.py", "from governance import kernel\n")
    assert "T1" in _rule_codes(lint_repo(fake_repo))


def test_t1_allows_core_contracts(fake_repo: Path):
    _write(
        fake_repo,
        "mind/fast_execute.py",
        "from core.contracts.events import SignalEvent\n",
    )
    assert "T1" not in _rule_codes(lint_repo(fake_repo))


# ---------------------------------------------------------------------------
# C2 / C3 / W1
# ---------------------------------------------------------------------------


def test_c2_fires_on_neuromorphic_to_execution(fake_repo: Path):
    _write(
        fake_repo,
        "mind/neuromorphic/decide.py",
        "import execution_engine\n",
    )
    assert "C2" in _rule_codes(lint_repo(fake_repo))


def test_c3_fires_on_web_autolearn_to_governance(fake_repo: Path):
    _write(
        fake_repo,
        "mind/web_autolearn/scrape.py",
        "import governance\n",
    )
    assert "C3" in _rule_codes(lint_repo(fake_repo))


def test_w1_fires_on_memecoin_to_main_wallet(fake_repo: Path):
    _write(
        fake_repo,
        "execution_engine/adapters/memecoin/router.py",
        "from wallet.main_wallet import resolve\n",
    )
    assert "W1" in _rule_codes(lint_repo(fake_repo))


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------


def test_l1_fires_learning_to_evolution(fake_repo: Path):
    _write(
        fake_repo,
        "learning_engine/lane.py",
        "from evolution_engine.engine import EvolutionEngine\n",
    )
    assert "L1" in _rule_codes(lint_repo(fake_repo))


def test_l1_fires_evolution_to_learning(fake_repo: Path):
    _write(
        fake_repo,
        "evolution_engine/lane.py",
        "from learning_engine.engine import LearningEngine\n",
    )
    assert "L1" in _rule_codes(lint_repo(fake_repo))


def test_l1_allows_shared_core_contracts(fake_repo: Path):
    _write(
        fake_repo,
        "learning_engine/lane.py",
        "from core.contracts.events import SystemEvent\n",
    )
    _write(
        fake_repo,
        "evolution_engine/lane.py",
        "from core.contracts.events import SystemEvent\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    assert "L1" not in codes


# ---------------------------------------------------------------------------
# L2 / L3
# ---------------------------------------------------------------------------


def test_l2_fires_offline_to_runtime(fake_repo: Path):
    _write(
        fake_repo,
        "learning_engine/bad.py",
        "from intelligence_engine.engine import IntelligenceEngine\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    assert "L2" in codes


def test_l2_allows_ledger_reader(fake_repo: Path):
    _write(
        fake_repo,
        "state/ledger/reader.py",
        "",
    )
    _write(
        fake_repo,
        "learning_engine/good.py",
        "from state.ledger.reader import LedgerReader\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    assert "L2" not in codes


def test_l3_fires_runtime_to_offline(fake_repo: Path):
    _write(
        fake_repo,
        "intelligence_engine/bad.py",
        "from learning_engine.engine import LearningEngine\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    assert "L3" in codes


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------


def test_b1_fires_intelligence_to_execution(fake_repo: Path):
    _write(
        fake_repo,
        "intelligence_engine/bad.py",
        "from execution_engine.engine import ExecutionEngine\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    assert "B1" in codes


def test_b1_allows_core_contracts(fake_repo: Path):
    _write(
        fake_repo,
        "intelligence_engine/good.py",
        "from core.contracts.events import SignalEvent\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    assert "B1" not in codes


def test_b1_fires_for_each_runtime_pair(fake_repo: Path):
    pairs = [
        ("intelligence_engine", "execution_engine"),
        ("intelligence_engine", "system_engine"),
        ("intelligence_engine", "governance_engine"),
        ("execution_engine", "system_engine"),
        ("execution_engine", "governance_engine"),
        ("system_engine", "governance_engine"),
    ]
    for src, dst in pairs:
        _write(
            fake_repo,
            f"{src}/bad_to_{dst}.py",
            f"from {dst}.engine import _\n",
        )
    codes = _rule_codes(lint_repo(fake_repo))
    assert codes.count("B1") >= len(pairs)


# ---------------------------------------------------------------------------
# B7 — dashboard isolation
# ---------------------------------------------------------------------------


def _scaffold_dashboard(fake_repo: Path) -> None:
    _write(fake_repo, "dashboard_backend/__init__.py", "")
    _write(fake_repo, "dashboard_backend/control_plane/__init__.py", "")


def test_b7_fires_on_dashboard_to_execution_engine(fake_repo: Path):
    _scaffold_dashboard(fake_repo)
    _write(
        fake_repo,
        "dashboard_backend/control_plane/bad.py",
        "from execution_engine.hot_path import run\n",
    )
    assert "B7" in _rule_codes(lint_repo(fake_repo))


def test_b7_fires_on_dashboard_to_learning_engine(fake_repo: Path):
    _scaffold_dashboard(fake_repo)
    _write(
        fake_repo,
        "dashboard_backend/control_plane/bad.py",
        "from learning_engine import smuggle\n",
    )
    assert "B7" in _rule_codes(lint_repo(fake_repo))


def test_b7_allows_core_contracts(fake_repo: Path):
    _scaffold_dashboard(fake_repo)
    _write(
        fake_repo,
        "dashboard_backend/control_plane/ok.py",
        "from core.contracts.events import SignalEvent\n",
    )
    assert "B7" not in _rule_codes(lint_repo(fake_repo))


def test_b7_allows_governance_control_plane(fake_repo: Path):
    _scaffold_dashboard(fake_repo)
    _write(
        fake_repo,
        "governance_engine/control_plane/__init__.py",
        "",
    )
    _write(
        fake_repo,
        "dashboard_backend/control_plane/ok.py",
        "from governance_engine.control_plane import OperatorInterfaceBridge\n",
    )
    assert "B7" not in _rule_codes(lint_repo(fake_repo))


def test_b7_allows_strategy_lifecycle_fsm(fake_repo: Path):
    _scaffold_dashboard(fake_repo)
    _write(
        fake_repo,
        "intelligence_engine/strategy_runtime/__init__.py",
        "",
    )
    _write(
        fake_repo,
        "intelligence_engine/strategy_runtime/state_machine.py",
        "",
    )
    _write(
        fake_repo,
        "dashboard_backend/control_plane/ok.py",
        "from intelligence_engine.strategy_runtime.state_machine "
        "import StrategyState\n",
    )
    assert "B7" not in _rule_codes(lint_repo(fake_repo))


# ---------------------------------------------------------------------------
# B17 — shadow meta-controller is non-acting (INV-52)
# ---------------------------------------------------------------------------


def test_b17_fires_on_shadow_to_governance(fake_repo: Path):
    _write(
        fake_repo,
        "intelligence_engine/meta_controller/__init__.py",
        "",
    )
    _write(
        fake_repo,
        "intelligence_engine/meta_controller/policy/__init__.py",
        "",
    )
    _write(
        fake_repo,
        "intelligence_engine/meta_controller/policy/shadow_policy.py",
        "from governance_engine import smuggle\n",
    )
    assert "B17" in _rule_codes(lint_repo(fake_repo))


# ---------------------------------------------------------------------------
# B20 — Triad Lock: Governance is order-blind (INV-56)
# ---------------------------------------------------------------------------


def test_b20_fires_on_governance_to_execution_engine(fake_repo: Path):
    _write(
        fake_repo,
        "governance_engine/bad.py",
        "from execution_engine.adapters.paper import PaperBroker\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    assert "B20" in codes


def test_b20_fires_on_governance_to_execution_hot_path(fake_repo: Path):
    _write(
        fake_repo,
        "governance_engine/bad.py",
        "from execution_engine.hot_path import fast_execute\n",
    )
    assert "B20" in _rule_codes(lint_repo(fake_repo))


def test_b20_does_not_fire_for_intelligence_to_execution(fake_repo: Path):
    # B1 still fires here, but B20 must not — B20 is governance-only.
    _write(
        fake_repo,
        "intelligence_engine/bad.py",
        "from execution_engine.engine import ExecutionEngine\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    assert "B20" not in codes


def test_b20_does_not_fire_for_governance_to_core_contracts(fake_repo: Path):
    _write(
        fake_repo,
        "governance_engine/ok.py",
        "from core.contracts.events import SystemEvent\n",
    )
    assert "B20" not in _rule_codes(lint_repo(fake_repo))


# ---------------------------------------------------------------------------
# B21 — Triad Lock: only execution_engine constructs ExecutionEvent (INV-56)
# ---------------------------------------------------------------------------


def test_b21_fires_on_governance_constructing_execution_event(
    fake_repo: Path,
):
    _write(
        fake_repo,
        "governance_engine/bad.py",
        "ExecutionEvent(ts_ns=0)\n",
    )
    assert "B21" in _rule_codes(lint_repo(fake_repo))


def test_b21_fires_on_intelligence_constructing_execution_event(
    fake_repo: Path,
):
    _write(
        fake_repo,
        "intelligence_engine/bad.py",
        "ExecutionEvent(ts_ns=0)\n",
    )
    assert "B21" in _rule_codes(lint_repo(fake_repo))


def test_b21_allows_execution_engine(fake_repo: Path):
    _write(
        fake_repo,
        "execution_engine/ok.py",
        "ExecutionEvent(ts_ns=0)\n",
    )
    assert "B21" not in _rule_codes(lint_repo(fake_repo))


def test_b21_allows_tests_directory(fake_repo: Path):
    _write(fake_repo, "tests/__init__.py", "")
    _write(
        fake_repo,
        "tests/test_thing.py",
        "ExecutionEvent(ts_ns=0)\n",
    )
    assert "B21" not in _rule_codes(lint_repo(fake_repo))


# ---------------------------------------------------------------------------
# B22 — Triad Lock: only intelligence_engine constructs SignalEvent (INV-56)
# ---------------------------------------------------------------------------


def test_b22_fires_on_governance_constructing_signal_event(
    fake_repo: Path,
):
    _write(
        fake_repo,
        "governance_engine/bad.py",
        "SignalEvent(ts_ns=0)\n",
    )
    assert "B22" in _rule_codes(lint_repo(fake_repo))


def test_b22_fires_on_execution_constructing_signal_event(fake_repo: Path):
    _write(
        fake_repo,
        "execution_engine/bad.py",
        "SignalEvent(ts_ns=0)\n",
    )
    assert "B22" in _rule_codes(lint_repo(fake_repo))


def test_b22_allows_intelligence_engine(fake_repo: Path):
    _write(
        fake_repo,
        "intelligence_engine/ok.py",
        "SignalEvent(ts_ns=0)\n",
    )
    assert "B22" not in _rule_codes(lint_repo(fake_repo))


def test_b22_allows_ui_dev_harness(fake_repo: Path):
    _write(fake_repo, "ui/__init__.py", "")
    _write(
        fake_repo,
        "ui/server.py",
        "SignalEvent(ts_ns=0)\n",
    )
    assert "B22" not in _rule_codes(lint_repo(fake_repo))


def test_b22_allows_tests_directory(fake_repo: Path):
    _write(fake_repo, "tests/__init__.py", "")
    _write(
        fake_repo,
        "tests/test_thing.py",
        "SignalEvent(ts_ns=0)\n",
    )
    assert "B22" not in _rule_codes(lint_repo(fake_repo))


# ---------------------------------------------------------------------------
# B31 — mode-effect table is the only mode-conditional decision oracle
# ---------------------------------------------------------------------------


def test_b31_fires_on_engine_hardcoding_mode(fake_repo: Path):
    _write(
        fake_repo,
        "intelligence_engine/regime_router.py",
        "from core.contracts.governance import SystemMode\n"
        "def gate(m):\n"
        "    return m == SystemMode.LIVE\n",
    )
    assert "B31" in _rule_codes(lint_repo(fake_repo))


def test_b31_allows_governance_control_plane(fake_repo: Path):
    _write(
        fake_repo,
        "governance_engine/control_plane/state_transition_manager.py",
        "from core.contracts.governance import SystemMode\n"
        "def init():\n"
        "    return SystemMode.LOCKED\n",
    )
    assert "B31" not in _rule_codes(lint_repo(fake_repo))


def test_b31_allows_mode_control_bar(fake_repo: Path):
    _write(fake_repo, "dashboard_backend/__init__.py", "")
    _write(fake_repo, "dashboard_backend/control_plane/__init__.py", "")
    _write(
        fake_repo,
        "dashboard_backend/control_plane/mode_control_bar.py",
        "from core.contracts.governance import SystemMode\n"
        "ALL = [SystemMode.SAFE, SystemMode.PAPER, SystemMode.LIVE]\n",
    )
    assert "B31" not in _rule_codes(lint_repo(fake_repo))


def test_b31_allows_tests(fake_repo: Path):
    _write(fake_repo, "tests/__init__.py", "")
    _write(
        fake_repo,
        "tests/test_modes.py",
        "from core.contracts.governance import SystemMode\n"
        "assert SystemMode.LIVE\n",
    )
    assert "B31" not in _rule_codes(lint_repo(fake_repo))


def test_b31_fires_on_execution_engine_set_membership(fake_repo: Path):
    _write(
        fake_repo,
        "execution_engine/dispatcher.py",
        "from core.contracts.governance import SystemMode\n"
        "ENABLED = {SystemMode.LIVE, SystemMode.CANARY}\n",
    )
    codes = _rule_codes(lint_repo(fake_repo))
    # Two attribute references → at least two B31 hits.
    assert codes.count("B31") >= 2
