#!/bin/bash

set -euo pipefail

BASE_DIR="$(readlink -f $(dirname $BASH_SOURCE))"
CUR_DIR="$(readlink -f $PWD)"

BUILD_DIR="$CUR_DIR/dir_build"
INSTALL_DIR="$CUR_DIR/dir_install"

mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"

cd "$BUILD_DIR"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL_DIR" "$BASE_DIR"
cmake --build . --target install --config Release/