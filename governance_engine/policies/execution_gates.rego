# ADAPTED FROM: open-policy-agent OPA policy evaluation contract
# (https://github.com/open-policy-agent/opa)
#
# A-07 baseline Rego policy: execution_gates.
#
# Decision semantics: block live order execution while the system is
# in SAFE mode. SAFE is the canonical "everything off" tier — paper
# brokers may still receive STAGE_ORDER actions, but EXECUTE_ORDER
# must be refused.
#
# The Python mirror lives in governance_engine/services/opa_policy.py
# (:func:`_execution_gate_blocked`).
#
# Input shape (mirrors PolicyInput.to_canonical_json()):
#
#     {
#       "action": "EXECUTE_ORDER",
#       "mode":   "SAFE" | "PAPER" | "CANARY" | "LIVE" | "AUTO" | "LOCKED",
#       "subject": "<operator-or-strategy-id>",
#       "payload": { ... }
#     }
#
# Output shape (consumed by _OpaClientTransport):
#
#     {
#       "result": true | false,
#       "metadata": {
#         "rejection_code": "POLICY_EXECUTION_GATE",
#         "summary":        "SAFE mode blocks order execution",
#         "rule_path":      "execution_gates/safe_blocks_orders"
#       }
#     }
package dix.governance.execution_gates

default allow := true

allow := false {
    input.action == "EXECUTE_ORDER"
    input.mode == "SAFE"
}

metadata := {
    "rejection_code": "POLICY_EXECUTION_GATE",
    "summary":        "SAFE mode blocks order execution",
    "rule_path":      "execution_gates/safe_blocks_orders",
}
