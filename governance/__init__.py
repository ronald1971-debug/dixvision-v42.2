"""governance -- patch pipeline + approvals + policy gates.

Every code change routes through `governance.patch_pipeline` (sandbox ->
authority-lint -> unit tests -> dep-scan -> shadow -> canary -> human
approval -> live) before being promoted. See patch_pipeline.py for details.
"""
