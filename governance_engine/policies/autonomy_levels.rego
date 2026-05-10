# ADAPTED FROM: open-policy-agent OPA policy evaluation contract
# (https://github.com/open-policy-agent/opa)
#
# A-07 baseline Rego policy: autonomy_levels.
#
# Decision semantics: escalate to the operator when the system is in
# AUTO and any hazard sensor is active. AUTO is the highest-autonomy
# tier — every active hazard must be acknowledged by a human before
# the request runs. The verdict is ESCALATE rather than REJECT so the
# operator-attention layer can pick up the request and prompt for an
# explicit ack.
#
# The Python mirror lives in governance_engine/services/opa_policy.py
# (:func:`_autonomy_level_escalation`).
#
# Input shape (mirrors PolicyInput.to_canonical_json()):
#
#     {
#       "action": "<any>",
#       "mode":   "SAFE" | "PAPER" | "CANARY" | "LIVE" | "AUTO" | "LOCKED",
#       "subject": "<operator-or-strategy-id>",
#       "payload": {
#         "active_hazards": int (number of currently-firing hazards)
#       }
#     }
#
# Output shape (consumed by _OpaClientTransport):
#
#     {
#       "result": "APPROVE" | "REJECT" | "ESCALATE",
#       "metadata": {
#         "rejection_code": "POLICY_AUTONOMY_ESCALATE",
#         "summary":        "AUTO mode requires zero active hazards",
#         "rule_path":      "autonomy_levels/auto_with_hazards"
#       }
#     }
package dix.governance.autonomy_levels

default allow := "APPROVE"

allow := "ESCALATE" {
    input.mode == "AUTO"
    input.payload.active_hazards > 0
}

metadata := {
    "rejection_code": "POLICY_AUTONOMY_ESCALATE",
    "summary":        "AUTO mode requires zero active hazards",
    "rule_path":      "autonomy_levels/auto_with_hazards",
}
