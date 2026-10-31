#!/usr/bin/env bash
set -euo pipefail

# auth in .pypirc
rm -rf ../viobench/dist
pip install --upgrade build twine
python3 -m build 
python3 -m twine upload dist/*
