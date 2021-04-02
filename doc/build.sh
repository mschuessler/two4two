#!/bin/bash

set -eu


die () { echo "ERROR: $*" >&2; exit 2; }

for cmd in pdoc; do
    command -v "$cmd" >/dev/null ||
        die "Missing $cmd; \`pip install $cmd\`"
done

BUILDROOT="doc/build"


echo
echo 'Building API reference docs'
echo "$BUILDROOT"

rm -r "$BUILDROOT" 2>/dev/null || true
mkdir -p "$BUILDROOT"

pdoc -d google \
      --output-dir "$BUILDROOT" \
      two4two
