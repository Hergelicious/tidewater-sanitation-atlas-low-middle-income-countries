#!/bin/bash
# Rewrite the absolute /home/claude root used in the scripts to another directory.
# Usage: bash set_root.sh /path/to/root   (inputs expected under <root>/data, outputs written under <root>/atlas)
set -e
ROOT="${1:?usage: set_root.sh /path/to/root}"
find . -name "*.py" -print0 | xargs -0 sed -i "s#/home/claude#${ROOT}#g"
echo "paths now point to ${ROOT}"
