#!/usr/bin/env bash
set -e
ARCH=$(uname -m)
PY_VER=$(python3 -c 'import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")')
WHL_DIR=$(mktemp -d)
trap "rm -rf $WHL_DIR" EXIT

cd "$WHL_DIR"
wget -q https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/aclruntime-0.0.2-cp${PY_VER}-cp${PY_VER}-linux_${ARCH}.whl
wget -q https://aisbench.obs.myhuaweicloud.com/packet/ais_bench_infer/0.0.2/ait/ais_bench-0.0.2-py3-none-any.whl

uv pip install *.whl
echo "[Success] install_for_cann"