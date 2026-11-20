#!/usr/bin/env bash
# Install Unitree's Python SDK (hardware only). Needs gcc/make and git; cmake comes from PyPI.
# cyclonedds==0.10.2 has no wheel for Python 3.12, so the CycloneDDS C library is built first.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m pip install -q cmake
mkdir -p third_party && cd third_party
[ -d cyclonedds ] || git clone --depth 1 -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git
../.venv/bin/cmake -S cyclonedds -B cyclonedds/build -DCMAKE_INSTALL_PREFIX="$PWD/cyclonedds/install" \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_IDLC_TESTING=OFF
../.venv/bin/cmake --build cyclonedds/build --target install -j"$(nproc)"
[ -d unitree_sdk2_python ] || git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git
export CYCLONEDDS_HOME="$PWD/cyclonedds/install"
../.venv/bin/pip install -e unitree_sdk2_python
../.venv/bin/python -c "from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_; print('unitree_sdk2py OK')"
