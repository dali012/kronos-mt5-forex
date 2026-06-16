#!/usr/bin/env bash
# Fetch the Kronos model code so `from model import ...` works, and pre-cache weights.
set -euo pipefail

VENDOR_DIR="$(dirname "$0")/../vendor"
mkdir -p "$VENDOR_DIR"

if [ ! -d "$VENDOR_DIR/Kronos" ]; then
  git clone https://github.com/shiyu-coder/Kronos.git "$VENDOR_DIR/Kronos"
fi

echo "Kronos cloned to $VENDOR_DIR/Kronos"
echo "Add it to PYTHONPATH so 'from model import Kronos, KronosTokenizer, KronosPredictor' resolves:"
echo "  export PYTHONPATH=\"$VENDOR_DIR/Kronos:\$PYTHONPATH\""
echo
echo "Models are pulled from Hugging Face on first use:"
echo "  tokenizer: NeoQuasar/Kronos-Tokenizer-base"
echo "  model:     NeoQuasar/Kronos-small"
