# ADAPTED FROM: open-policy-agent OPA policy evaluation contract
# (https://github.com/open-policy-agent/opa)
#
# A-07 baseline Rego policy: position_limits.
#
# Decision semantics: reject any request whose proposed notional
# exposure exceeds the configured per-symbol notional limit. The
# Python mirror lives in governance_engine/services/opa_policy.py
# (:func:`_position_limit_violated`); the two must remain in lockstep
# so the in-process InProcessPolicyTransport and the external OPA
# binary produce the same verdict for the same input.
#
# Input shape (mirrors PolicyInput.to_canonical_json()):
#
#     {
#       "action": "EXECUTE_ORDER" | "STAGE_ORDER" | ...,
#       "mode":   "SAFE" | "PAPER" | "CANARY" | "LIVE" | "AUTO" | "LOCKED",
#       "subject": "<operator-or-strategy-id>",
#       "payload": {
#         "notional_usd":       float,
#         "notional_limit_usd": float
#       }
#     }
#
# Output shape (consumed by _OpaClientTransport):
#
#     {
#       "result": true | false | "APPROVE" | "REJECT" | "ESCALATE",
#       "metadata": {
#         "rejection_code": "POLICY_POSITION_LIMIT",
#         "summary":        "notional exceeds configured limit",
#         "rule_path":      "position_limits/notional_exceeded"
#       }
#     }
package dix.governance.position_limits

default allow := true

# Reject when notional > limit. Missing keys mean the rule does not
# fire — that is intentional: the caller is responsible for supplying
# both fields, and a missing field is handled by an upstream
# request-shape check rather than this rule.
allow := false {
    input.payload.notional_usd > input.payload.notional_limit_usd
}

metadata := {
    "rejection_code": "POLICY_POSITION_LIMIT",
    "summary":        "notional exceeds configured limit",
    "rule_path":      "position_limits/notional_exceeded",
}
