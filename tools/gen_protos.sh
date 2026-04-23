#!/usr/bin/env bash
# tools/gen_protos.sh
#
# Generate cross-language stubs for every contract in contracts/*.proto.
# Python stubs land in contracts/gen/python/; other languages will be
# added as their toolchains arrive (Rust via prost-build in a Cargo
# build.rs; Go via protoc-gen-go in the Go module).
#
# Generated files are NOT checked in (see contracts/README.md and
# .gitignore). CI re-runs this script on every build and fails if the
# checked-in .proto files do not parse.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACTS="${REPO_ROOT}/contracts"
OUT_PY="${CONTRACTS}/gen/python"

mkdir -p "${OUT_PY}"
touch "${OUT_PY}/__init__.py"

# Regenerate Python stubs for every .proto file.
python3 -m grpc_tools.protoc \
    --proto_path="${CONTRACTS}" \
    --python_out="${OUT_PY}" \
    "${CONTRACTS}"/*.proto

echo "[gen_protos] Python stubs written to ${OUT_PY}"
