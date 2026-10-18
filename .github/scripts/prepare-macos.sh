#!/bin/bash
set -euo pipefail

# Build liblzma ourselves: Homebrew bottles target the runner OS, not macOS 14.
# The workflow checks out the immutable XZ 5.8.1 commit into .ci-xz.
prefix="$RUNNER_TEMP/kenlm-deps"
cmake -S "$GITHUB_WORKSPACE/.ci-xz" -B "$RUNNER_TEMP/xz-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF \
  -DXZ_NLS=OFF -DXZ_DOC=OFF \
  -DXZ_TOOL_XZ=OFF -DXZ_TOOL_XZDEC=OFF \
  -DXZ_TOOL_LZMADEC=OFF -DXZ_TOOL_LZMAINFO=OFF
cmake --build "$RUNNER_TEMP/xz-build" --parallel 3
cmake --install "$RUNNER_TEMP/xz-build"

# The SDK supplies zlib and bzip2. Exclude preinstalled Homebrew libraries.
{
  echo "CPPFLAGS=-I$prefix/include"
  echo "CXXFLAGS=-I$prefix/include"
  echo "LDFLAGS=-L$prefix/lib"
  echo "CMAKE_ARGS=-DCMAKE_PREFIX_PATH=$prefix -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew;/usr/local"
  echo "PIP_CONSTRAINT=$GITHUB_WORKSPACE/.github/constraints/macos-build.txt"
  echo "SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD)"
} >> "$GITHUB_ENV"
