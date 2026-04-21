"""
scripts/generate_hash.py
DIX VISION v42.2 — Foundation Hash Generator

Run after any immutable_core changes:
  python scripts/generate_hash.py
"""
import hashlib
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent.parent
    foundation = root / "immutable_core" / "foundation.py"
    if not foundation.exists():
        print(f"ERROR: {foundation} not found")
        sys.exit(1)
    hash_val = hashlib.sha256(foundation.read_bytes()).hexdigest()
    hash_path = root / "immutable_core" / "foundation.hash"
    hash_path.write_text(hash_val + "\n")
    print(f"Foundation hash written: {hash_val}")
    print(f"File: {hash_path}")

if __name__ == "__main__":
    main()
