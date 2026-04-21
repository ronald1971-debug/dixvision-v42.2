import glob
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
print("Project root added to sys.path")
print("Current dir:", os.getcwd())
print("Immutable core files:", glob.glob("immutable_core/*.py"))

try:
    from immutable_core.foundation import (
        EXPECTED_FOUNDATION_HASH,
        get_current_foundation_hash,
        verify_foundation,
    )
    current = get_current_foundation_hash()
    print("Import successful")
    print("Current computed hash :", current)
    print("Expected hash        :", EXPECTED_FOUNDATION_HASH)
    print("Match?               :", current == EXPECTED_FOUNDATION_HASH)
    verify_foundation()
except Exception as e:
    print("ERROR type :", type(e).__name__)
    print("ERROR message :", str(e))
    traceback.print_exc()
