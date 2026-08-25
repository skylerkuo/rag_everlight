#!/usr/bin/env bash
set -euo pipefail

cd /home/gai/Desktop/rag_multisource_system_v6
source .venv/bin/activate
python rag.py build --context-radius 1
