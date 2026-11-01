#!/bin/sh
# Cross-build gurt.dll (64-bit Windows) from Linux / WSL / macOS with Zig's bundled
# MinGW toolchain -- no Visual Studio needed.  Output: prebuilt/windows-x64/gurt.dll
#   pip install ziglang
set -e
cd "$(dirname "$0")"
mkdir -p prebuilt/windows-x64
python3 -m ziglang cc -target x86_64-windows-gnu -O2 -std=c99 -Wall -Iinclude -shared \
    src/despeckle.c src/deglitch.c src/gapfill.c -o prebuilt/windows-x64/gurt.dll
rm -f prebuilt/windows-x64/*.lib prebuilt/windows-x64/*.pdb
echo "built prebuilt/windows-x64/gurt.dll"
